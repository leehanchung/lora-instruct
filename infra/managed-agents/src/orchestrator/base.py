"""Abstract base for session orchestrators.

Supports two backends:
  - K8s Jobs (original): each session = a pod. Simple, but ~2-5s startup overhead.
  - Node Agent (hybrid): control plane in K8s, sessions as bubblewrap-sandboxed
    processes on worker nodes. Near-instant startup, closer to Anthropic's actual
    architecture.

Workers are heterogeneous — different hardware profiles, capabilities, and
resource availability. The scheduler does constraint-based bin-packing rather
than simple load balancing.
"""

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional


class SessionStatus(str, Enum):
    """Unified session status across orchestrator backends."""

    PENDING = "pending"  # Scheduled but not yet running
    RUNNING = "running"  # Agent loop is active
    IDLE = "idle"  # Session alive but no active turn
    TERMINATED = "terminated"  # Finished or failed
    UNKNOWN = "unknown"


class SchedulingStrategy(str, Enum):
    """How the scheduler places sessions on workers."""

    BIN_PACK = "bin_pack"  # Pack tightly — maximize free capacity on other nodes
    SPREAD = "spread"  # Spread across nodes — maximize fault tolerance
    RANDOM = "random"  # Random among eligible — simplest, decent for mixed loads


class SandboxBackend(str, Enum):
    """Isolation technology for session sandboxing.

    Tradeoff spectrum (weakest/fastest → strongest/slowest):
      BUBBLEWRAP: Linux namespaces + seccomp. ~0ms overhead. Real kernel.
                  Good for: trusted agent code, prompt injection defense.
      GVISOR:     User-space kernel (Sentry). ~5-15% syscall overhead.
                  Good for: untrusted tool code, multi-tenant SaaS.
      FIRECRACKER: MicroVM with dedicated kernel. ~125ms boot.
                  Good for: full tenant isolation, compliance requirements.
    """

    BUBBLEWRAP = "bubblewrap"  # Default — namespace isolation, near-zero overhead
    GVISOR = "gvisor"  # Stronger — user-space kernel, opt-in per session
    FIRECRACKER = "firecracker"  # Strongest — microVM, requires KVM


# ---------------------------------------------------------------------------
# Resource & capability model
# ---------------------------------------------------------------------------

@dataclass
class ResourceRequest:
    """What a session needs from a worker.

    Hard constraints MUST be satisfied (session won't start otherwise).
    Soft preferences are used for ranking among eligible workers.
    """

    # Quantitative resources (hard floor — worker must have at least this much free)
    cpu_millicores: int = 1000
    memory_mb: int = 2048
    gpu_count: int = 0
    disk_mb: int = 1024  # Workspace storage

    # Hard constraints: worker MUST have all of these labels
    # e.g. {"gpu_type": "a100", "arch": "amd64", "pool": "inference"}
    required_labels: dict[str, str] = field(default_factory=dict)

    # Soft preferences: worker with these labels is preferred but not required
    # e.g. {"storage": "ssd", "region": "us-west-2"}
    preferred_labels: dict[str, str] = field(default_factory=dict)

    # Anti-affinity: avoid workers running sessions with these tags
    # e.g. ["user:alice"] means don't colocate with other sessions from alice
    anti_affinity_tags: list[str] = field(default_factory=list)

    # Affinity: prefer workers running sessions with these tags
    # e.g. ["model:llama-3.1-8b"] for shared model cache locality
    affinity_tags: list[str] = field(default_factory=list)

    # Priority: higher priority sessions can preempt lower ones (0=default)
    priority: int = 0

    # Maximum time the session is allowed to run (seconds)
    timeout_seconds: int = 3600

    # Sandbox isolation backend (default: bubblewrap)
    sandbox_backend: str = "bubblewrap"  # "bubblewrap", "gvisor", or "firecracker"


@dataclass
class SessionInfo:
    """Snapshot of a running session."""

    session_id: str
    status: SessionStatus
    worker_id: Optional[str] = None  # Which worker node hosts this session
    pid: Optional[int] = None  # Process ID (node-agent backend only)
    resources: Optional[ResourceRequest] = None  # What was allocated
    tags: list[str] = field(default_factory=list)  # Tags for affinity matching
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkerCapabilities:
    """What a worker can offer — reported by the node agent."""

    # Total hardware resources
    total_cpu_millicores: int = 4000
    total_memory_mb: int = 8192
    total_gpu_count: int = 0
    total_disk_mb: int = 50000

    # Currently available (total minus what's allocated to running sessions)
    available_cpu_millicores: int = 4000
    available_memory_mb: int = 8192
    available_gpu_count: int = 0
    available_disk_mb: int = 50000

    # Worker labels (capabilities, hardware profile, pool membership)
    # e.g. {"arch": "amd64", "gpu_type": "a100", "pool": "inference",
    #        "storage": "ssd", "region": "us-west-2"}
    labels: dict[str, str] = field(default_factory=dict)

    # Tags of currently running sessions (for affinity/anti-affinity)
    active_session_tags: list[str] = field(default_factory=list)


@dataclass
class WorkerInfo:
    """Full worker state: identity + capabilities."""

    worker_id: str
    address: str  # host:port for the node agent HTTP endpoint
    capabilities: WorkerCapabilities = field(default_factory=WorkerCapabilities)
    active_sessions: int = 0
    max_sessions: int = 20
    healthy: bool = True
    last_heartbeat_epoch: float = 0.0


class BaseOrchestrator(abc.ABC):
    """Interface that both K8s and node-agent orchestrators implement."""

    @abc.abstractmethod
    async def create_session(
        self,
        session_id: str,
        agent_config: dict,
        environment_config: dict,
        redis_url: str,
        api_base_url: str,
        resources: Optional[ResourceRequest] = None,
        tags: Optional[list[str]] = None,
    ) -> SessionInfo:
        """Spin up a new agent session.

        Args:
            session_id: Unique session identifier.
            agent_config: Agent configuration (model, tools, system prompt, etc.).
            environment_config: Environment configuration (packages, resources, networking).
            redis_url: Redis connection URL for event pub/sub.
            api_base_url: Base URL for API callbacks from the agent loop.
            resources: Resource requirements and scheduling constraints.
            tags: Tags for this session (used in affinity/anti-affinity matching).

        Returns:
            SessionInfo with initial status.
        """
        ...

    @abc.abstractmethod
    async def terminate_session(self, session_id: str) -> None:
        """Kill a session and clean up resources."""
        ...

    @abc.abstractmethod
    async def get_session_status(self, session_id: str) -> SessionInfo:
        """Query current status of a session."""
        ...

    @abc.abstractmethod
    async def stream_logs(self, session_id: str) -> AsyncIterator[str]:
        """Stream stdout/stderr from a session for debugging."""
        ...

    async def list_workers(self) -> list[WorkerInfo]:
        """List available workers and their capacity.

        Only meaningful for the node-agent backend. K8s backend returns empty.
        """
        return []

    async def close(self) -> None:
        """Clean up connections."""
        pass


__all__ = [
    "BaseOrchestrator",
    "ResourceRequest",
    "SchedulingStrategy",
    "SessionInfo",
    "SessionStatus",
    "WorkerCapabilities",
    "WorkerInfo",
]
