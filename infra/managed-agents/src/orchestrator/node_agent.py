"""Node agent: runs on each worker, manages Firecracker microVM sessions.

Each session is a Firecracker microVM running a harness (Claude Code,
SWE-agent, custom Python, etc.). The node agent:
  - Creates/boots/terminates VMs via FirecrackerSandbox
  - Reports worker capabilities (CPU, GPU, memory, disk, labels)
  - Reports active session tags (for scheduler affinity/anti-affinity)
  - Periodically cleans up dead VMs

Runs as either:
  - K8s DaemonSet (hybrid mode)
  - systemd service on bare metal
"""

import asyncio
import json
import logging
import os
import platform
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .harness import BUILTIN_HARNESSES, HarnessDefinition, SandboxRuntime
from .sandbox import FirecrackerSandbox, SandboxConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class HarnessPayload(BaseModel):
    """Harness config sent by the scheduler."""

    name: str  # Must match a built-in or be a full custom definition
    rootfs_path: str = ""  # Path to rootfs on this worker
    image: str = ""  # OCI image (for rootfs build or container fallback)
    entrypoint: list[str] = []
    command: list[str] = []
    runtime: str = "firecracker"  # "firecracker", "gvisor", "runc"


class ResourceRequestPayload(BaseModel):
    vcpus: int = 2
    memory_mb: int = 2048
    disk_mb: int = 4096
    gpu_count: int = 0
    timeout_seconds: int = 3600

    # Backward compat
    cpu_millicores: int = 0  # If set, converted to vcpus

    def effective_vcpus(self) -> int:
        if self.cpu_millicores > 0:
            return max(1, self.cpu_millicores // 1000)
        return self.vcpus


class CreateSessionRequest(BaseModel):
    session_id: str
    harness: HarnessPayload
    env: dict[str, str] = Field(default_factory=dict)
    resource_overrides: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Legacy fields (backward compat with hybrid orchestrator)
    agent_config: dict = Field(default_factory=dict)
    environment_config: dict = Field(default_factory=dict)
    redis_url: str = ""
    api_base_url: str = ""
    resource_request: ResourceRequestPayload = Field(
        default_factory=ResourceRequestPayload
    )


class SessionResponse(BaseModel):
    session_id: str
    vm_id: str
    harness_name: str
    status: str
    pid: int | None = None
    tags: list[str] = []
    runtime: str = "firecracker"


class CapabilitiesResponse(BaseModel):
    total_cpu_millicores: int
    total_memory_mb: int
    total_gpu_count: int
    total_disk_mb: int
    available_cpu_millicores: int
    available_memory_mb: int
    available_gpu_count: int
    available_disk_mb: int
    labels: dict[str, str]
    active_session_tags: list[str]
    available_runtimes: list[str]
    available_harnesses: list[str]
    available_rootfs: list[str]  # rootfs images present on this worker


class NodeHealthResponse(BaseModel):
    worker_id: str
    hostname: str
    arch: str
    capabilities: CapabilitiesResponse
    active_sessions: int
    max_sessions: int
    healthy: bool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKER_ID = os.getenv("WORKER_ID", platform.node())
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "20"))
WORKSPACES_ROOT = os.getenv("WORKSPACES_ROOT", "/var/lib/managed-agents/workspaces")
LOGS_ROOT = os.getenv("LOGS_ROOT", "/var/lib/managed-agents/logs")
SOCKETS_ROOT = os.getenv("SOCKETS_ROOT", "/var/lib/managed-agents/sockets")
ROOTFS_ROOT = os.getenv("ROOTFS_ROOT", "/var/lib/managed-agents/rootfs")
KERNEL_PATH = os.getenv(
    "FC_KERNEL_PATH", "/var/lib/managed-agents/kernel/vmlinux"
)

# Worker labels
WORKER_LABELS: dict[str, str] = {}
labels_str = os.getenv("WORKER_LABELS", "")
if labels_str:
    for pair in labels_str.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            WORKER_LABELS[k.strip()] = v.strip()

# Custom harness definitions (loaded from JSON file or env)
CUSTOM_HARNESSES_PATH = os.getenv("CUSTOM_HARNESSES_PATH", "")

sandbox: FirecrackerSandbox = None  # type: ignore[assignment]
_session_tags: dict[str, list[str]] = {}
_session_harness: dict[str, str] = {}  # session_id → harness name
_allocated_resources: dict[str, ResourceRequestPayload] = {}
_all_harnesses: dict[str, HarnessDefinition] = dict(BUILTIN_HARNESSES)


def _load_custom_harnesses() -> None:
    """Load custom harness definitions from JSON file."""
    if not CUSTOM_HARNESSES_PATH or not Path(CUSTOM_HARNESSES_PATH).exists():
        return
    try:
        with open(CUSTOM_HARNESSES_PATH) as f:
            custom = json.load(f)
        for name, defn in custom.items():
            _all_harnesses[name] = HarnessDefinition(name=name, **defn)
        logger.info(f"Loaded {len(custom)} custom harnesses from {CUSTOM_HARNESSES_PATH}")
    except Exception as e:
        logger.error(f"Failed to load custom harnesses: {e}")


def _detect_available_rootfs() -> list[str]:
    """Scan ROOTFS_ROOT for available ext4 rootfs images."""
    rootfs_dir = Path(ROOTFS_ROOT)
    if not rootfs_dir.exists():
        return []
    return [
        f.stem  # e.g. "claude-code" from "claude-code.ext4"
        for f in rootfs_dir.glob("*.ext4")
    ]


def _resolve_harness(payload: HarnessPayload) -> HarnessDefinition:
    """Resolve a harness payload to a full HarnessDefinition."""
    # Try built-in / registered harnesses first
    if payload.name in _all_harnesses:
        harness = _all_harnesses[payload.name]
        # Apply overrides from payload
        overrides = {}
        if payload.rootfs_path:
            overrides["rootfs_path"] = payload.rootfs_path
        if payload.image:
            overrides["image"] = payload.image
        if payload.entrypoint:
            overrides["entrypoint"] = payload.entrypoint
        if payload.command:
            overrides["command"] = payload.command
        if payload.runtime != "firecracker":
            overrides["runtime"] = SandboxRuntime(payload.runtime)
        if overrides:
            harness = _clone_harness(harness, **overrides)
        return harness

    # Unknown harness — build from payload (must have rootfs or image)
    if not payload.rootfs_path and not payload.image:
        raise ValueError(
            f"Unknown harness '{payload.name}' and no rootfs_path or image provided"
        )
    return HarnessDefinition(
        name=payload.name,
        rootfs_path=payload.rootfs_path,
        image=payload.image,
        entrypoint=payload.entrypoint,
        command=payload.command,
        runtime=SandboxRuntime(payload.runtime),
    )


def _clone_harness(base: HarnessDefinition, **overrides) -> HarnessDefinition:
    """Create a copy of a harness with specific fields overridden."""
    from dataclasses import asdict, fields as dc_fields

    data = {}
    for f in dc_fields(base):
        data[f.name] = getattr(base, f.name)
    data.update(overrides)

    # Re-create nested objects
    if "network" in data and isinstance(data["network"], dict):
        from .harness import HarnessNetworkPolicy
        data["network"] = HarnessNetworkPolicy(**data["network"])
    if "runtime" in data and isinstance(data["runtime"], str):
        data["runtime"] = SandboxRuntime(data["runtime"])
    return HarnessDefinition(**data)


def _harness_to_sandbox_config(
    harness: HarnessDefinition,
    session_env: dict[str, str],
    resource_overrides: dict,
) -> SandboxConfig:
    """Convert a HarnessDefinition + session env into a SandboxConfig for Firecracker."""

    # Resolve rootfs path — check worker-local rootfs directory first
    rootfs_path = harness.rootfs_path
    if not rootfs_path or not Path(rootfs_path).exists():
        # Try convention: ROOTFS_ROOT/<harness-name>.ext4
        candidate = Path(ROOTFS_ROOT) / f"{harness.name}.ext4"
        if candidate.exists():
            rootfs_path = str(candidate)
        else:
            raise FileNotFoundError(
                f"Rootfs not found for harness '{harness.name}'. "
                f"Checked: {harness.rootfs_path}, {candidate}. "
                f"Build it with: python -m src.orchestrator.rootfs_builder --harness {harness.name}"
            )

    # Merge harness base_env with session-specific env
    merged_env = dict(harness.base_env)
    merged_env.update(session_env)

    # Apply resource overrides
    vcpus = resource_overrides.get("vcpus", harness.default_vcpus)
    memory_mb = resource_overrides.get("memory_mb", harness.default_memory_mb)
    disk_mb = resource_overrides.get("disk_mb", harness.default_disk_mb)
    timeout = resource_overrides.get("timeout_seconds", harness.default_timeout_seconds)

    # Backward compat: cpu_millicores → vcpus
    if "cpu_millicores" in resource_overrides:
        vcpus = max(1, resource_overrides["cpu_millicores"] // 1000)

    return SandboxConfig(
        vcpu_count=vcpus,
        memory_mb=memory_mb,
        disk_size_mb=disk_mb,
        allow_network=harness.network.enabled,
        allowed_domains=harness.network.resolved_domains(),
        kernel_path=KERNEL_PATH,
        rootfs_path=rootfs_path,
        timeout_seconds=timeout,
        env=merged_env,
        entrypoint=harness.entrypoint or None,
        use_jailer=harness.use_jailer,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sandbox
    sandbox = FirecrackerSandbox(
        workspaces_root=WORKSPACES_ROOT,
        logs_root=LOGS_ROOT,
        sockets_root=SOCKETS_ROOT,
    )
    _load_custom_harnesses()
    logger.info(
        f"Node agent started: worker_id={WORKER_ID}, max_sessions={MAX_SESSIONS}, "
        f"labels={WORKER_LABELS}, harnesses={list(_all_harnesses.keys())}, "
        f"rootfs={_detect_available_rootfs()}"
    )

    # Background cleanup task
    async def cleanup_loop():
        while True:
            await asyncio.sleep(60)
            cleaned = await sandbox.cleanup_dead()
            # Also clean our tracking dicts
            active = set(sandbox.active_sessions().keys())
            for sid in list(_session_tags.keys()):
                if sid not in active:
                    _session_tags.pop(sid, None)
                    _session_harness.pop(sid, None)
                    _allocated_resources.pop(sid, None)

    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    cleanup_task.cancel()
    for sid in list(sandbox.active_sessions().keys()):
        await sandbox.terminate_session(sid)
    logger.info("Node agent shut down")


app = FastAPI(
    title="Managed Agents Node Agent",
    version="0.4.0",
    description="Worker-level agent that manages Firecracker microVM sandboxes",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(req: CreateSessionRequest):
    """Start a new microVM session."""
    if sandbox.active_session_count() >= MAX_SESSIONS:
        raise HTTPException(503, "Worker at capacity")

    # Resolve harness
    try:
        harness = _resolve_harness(req.harness)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    # Merge env: legacy fields + explicit env
    session_env = dict(req.env)
    if req.redis_url:
        session_env.setdefault("REDIS_URL", req.redis_url)
    if req.api_base_url:
        session_env.setdefault("API_BASE_URL", req.api_base_url)

    # Resource overrides
    overrides = dict(req.resource_overrides)
    if not overrides and req.resource_request:
        overrides = {
            "vcpus": req.resource_request.effective_vcpus(),
            "memory_mb": req.resource_request.memory_mb,
            "disk_mb": req.resource_request.disk_mb,
            "timeout_seconds": req.resource_request.timeout_seconds,
        }

    # Build sandbox config from harness + env + overrides
    try:
        config = _harness_to_sandbox_config(harness, session_env, overrides)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e

    # Boot the microVM
    try:
        info = await sandbox.create_session(
            session_id=req.session_id,
            config=config,
        )
    except Exception as e:
        logger.error(f"Failed to create session {req.session_id}: {e}")
        raise HTTPException(500, f"Failed to create session: {e}") from e

    _session_tags[req.session_id] = req.tags
    _session_harness[req.session_id] = harness.name
    _allocated_resources[req.session_id] = req.resource_request

    return SessionResponse(
        session_id=req.session_id,
        vm_id=info.vm_id,
        harness_name=harness.name,
        status="running",
        pid=info.pid,
        tags=req.tags,
        runtime=harness.runtime.value,
    )


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    await sandbox.terminate_session(session_id)
    _session_tags.pop(session_id, None)
    _session_harness.pop(session_id, None)
    _allocated_resources.pop(session_id, None)


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    info = await sandbox.get_session_info(session_id)
    if info is None:
        raise HTTPException(404, "Session not found or terminated")

    status = await sandbox.get_session_status(session_id)

    return SessionResponse(
        session_id=session_id,
        vm_id=info.vm_id,
        harness_name=_session_harness.get(session_id, "unknown"),
        status=status,
        pid=info.pid,
        tags=_session_tags.get(session_id, []),
    )


@app.get("/sessions", response_model=list[SessionResponse])
async def list_sessions():
    result = []
    for sid, info in sandbox.active_sessions().items():
        status = await sandbox.get_session_status(sid)
        result.append(
            SessionResponse(
                session_id=sid,
                vm_id=info.vm_id,
                harness_name=_session_harness.get(sid, "unknown"),
                status=status,
                pid=info.pid,
                tags=_session_tags.get(sid, []),
            )
        )
    return result


@app.get("/sessions/{session_id}/logs")
async def stream_session_logs(session_id: str):
    info = await sandbox.get_session_info(session_id)
    if info is None:
        raise HTTPException(404, "Session not found")

    return StreamingResponse(
        sandbox.stream_logs(session_id),
        media_type="text/plain",
    )


@app.get("/harnesses")
async def list_harnesses():
    """List all available harnesses on this worker."""
    available_rootfs = set(_detect_available_rootfs())
    return {
        name: {
            "description": h.description,
            "rootfs_path": h.rootfs_path,
            "image": h.image,
            "runtime": h.runtime.value,
            "default_vcpus": h.default_vcpus,
            "default_memory": h.default_memory_mb,
            "rootfs_available": h.name in available_rootfs,
        }
        for name, h in _all_harnesses.items()
    }


@app.get("/health", response_model=NodeHealthResponse)
async def health():
    mem = psutil.virtual_memory()
    cpu_count = psutil.cpu_count() or 1
    disk = shutil.disk_usage(WORKSPACES_ROOT)

    # Calculate allocated resources from active sessions
    allocated_vcpus = sum(r.effective_vcpus() for r in _allocated_resources.values())
    allocated_mem = sum(r.memory_mb for r in _allocated_resources.values())
    allocated_gpu = sum(r.gpu_count for r in _allocated_resources.values())

    total_cpu = cpu_count * 1000  # Report in millicores for scheduler compat
    total_mem = int(mem.total / (1024 * 1024))
    total_gpu = _detect_gpu_count()
    total_disk = int(disk.total / (1024 * 1024))

    all_tags = []
    for tags in _session_tags.values():
        all_tags.extend(tags)

    return NodeHealthResponse(
        worker_id=WORKER_ID,
        hostname=platform.node(),
        arch=platform.machine(),
        capabilities=CapabilitiesResponse(
            total_cpu_millicores=total_cpu,
            total_memory_mb=total_mem,
            total_gpu_count=total_gpu,
            total_disk_mb=total_disk,
            available_cpu_millicores=max(0, total_cpu - allocated_vcpus * 1000),
            available_memory_mb=max(0, total_mem - allocated_mem),
            available_gpu_count=max(0, total_gpu - allocated_gpu),
            available_disk_mb=int(disk.free / (1024 * 1024)),
            labels=dict(WORKER_LABELS),
            active_session_tags=all_tags,
            available_runtimes=_detect_runtimes(),
            available_harnesses=list(_all_harnesses.keys()),
            available_rootfs=_detect_available_rootfs(),
        ),
        active_sessions=sandbox.active_session_count(),
        max_sessions=MAX_SESSIONS,
        healthy=True,
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_gpu_count() -> int:
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _detect_runtimes() -> list[str]:
    """Detect which sandbox runtimes are available on this worker."""
    runtimes = []
    # Firecracker
    if shutil.which("firecracker"):
        runtimes.append("firecracker")
    # gVisor
    if shutil.which("runsc"):
        runtimes.append("gvisor")
    # Standard containers (fallback)
    if shutil.which("runc"):
        runtimes.append("runc")
    return runtimes or ["firecracker"]  # Assume FC if detection fails


__all__ = ["app"]
