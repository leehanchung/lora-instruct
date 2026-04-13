"""Firecracker microVM sandbox manager.

Each agent session runs in its own Firecracker microVM:
  - Own Linux kernel (no shared kernel attack surface)
  - Own root filesystem (from harness rootfs image)
  - Own network namespace (tap device per VM)
  - Own memory (ballooning for overcommit)
  - ~125ms boot, ~5MB overhead per VM

Architecture per sandbox:

  ┌─────────────────────────────────┐
  │  Firecracker microVM            │
  │                                 │
  │  Linux kernel (5.10 minimal)    │
  │  ┌───────────────────────────┐  │
  │  │ rootfs (ext4, from image) │  │
  │  │                           │  │
  │  │  /usr/bin/claude          │  │  ← harness binary
  │  │  /workspace/              │  │  ← writable workspace
  │  │  /etc/agent-config.json   │  │  ← session config
  │  └───────────────────────────┘  │
  │                                 │
  │  virtio-net  ← tap device       │  ← filtered network
  │  virtio-vsock ← host↔VM comms   │  ← control channel
  └─────────────────────────────────┘

The node agent talks to Firecracker via its REST API (Unix socket per VM),
and to the VM's init process via vsock for health checks and log streaming.

Rootfs images are built per harness type (see rootfs_builder.py).
"""

import asyncio
import json
import logging
import os
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# Firecracker binary paths (install via https://github.com/firecracker-microvm/firecracker)
DEFAULT_FC_BIN = "/usr/local/bin/firecracker"
DEFAULT_JAILER_BIN = "/usr/local/bin/jailer"
DEFAULT_KERNEL_PATH = "/var/lib/managed-agents/kernel/vmlinux"


@dataclass
class SandboxConfig:
    """Configuration for a Firecracker microVM sandbox."""

    # VM resources
    vcpu_count: int = 2
    memory_mb: int = 2048
    disk_size_mb: int = 4096  # Workspace overlay size

    # Networking
    allow_network: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    tap_device_prefix: str = "fc-"  # tap device naming

    # Paths
    kernel_path: str = DEFAULT_KERNEL_PATH
    rootfs_path: str = ""  # Path to harness rootfs image (ext4)

    # Timeout
    timeout_seconds: int = 3600  # 0 = no timeout

    # Environment variables to inject into the VM
    env: dict[str, str] = field(default_factory=dict)

    # Entrypoint override (if different from rootfs default)
    entrypoint: Optional[list[str]] = None

    # Jailer config (extra isolation)
    use_jailer: bool = True  # Run Firecracker inside jailer for defense in depth
    jailer_uid: int = 1000
    jailer_gid: int = 1000


@dataclass
class MicroVMInfo:
    """Handle to a running Firecracker microVM."""

    session_id: str
    vm_id: str  # Short ID for Firecracker
    pid: Optional[int]  # Firecracker process PID
    api_socket: Path  # Unix socket for Firecracker API
    vsock_path: Path  # vsock for host↔VM communication
    workspace_path: Path  # Host path to workspace overlay
    log_path: Path  # VM console/serial log
    tap_device: Optional[str]  # Network tap device name
    config: SandboxConfig
    process: Optional[asyncio.subprocess.Process] = None


class FirecrackerSandbox:
    """Manages Firecracker microVM sandboxes on a single worker.

    Lifecycle per session:
      1. Prepare workspace overlay (writable layer on top of rootfs)
      2. Configure network (create tap device, attach to bridge)
      3. Start Firecracker with VM config (kernel, rootfs, drives, network)
      4. VM boots, init starts the harness entrypoint
      5. Host communicates via vsock (health, logs, control)
      6. On terminate: send CtrlAltDel, wait, then kill process
    """

    def __init__(
        self,
        workspaces_root: str = "/var/lib/managed-agents/workspaces",
        logs_root: str = "/var/lib/managed-agents/logs",
        sockets_root: str = "/var/lib/managed-agents/sockets",
        fc_bin: str = DEFAULT_FC_BIN,
        jailer_bin: str = DEFAULT_JAILER_BIN,
    ) -> None:
        self.workspaces_root = Path(workspaces_root)
        self.logs_root = Path(logs_root)
        self.sockets_root = Path(sockets_root)
        self.fc_bin = fc_bin
        self.jailer_bin = jailer_bin
        self._vms: dict[str, MicroVMInfo] = {}

        # Ensure directories exist
        for d in [self.workspaces_root, self.logs_root, self.sockets_root]:
            d.mkdir(parents=True, exist_ok=True)

    async def create_session(
        self,
        session_id: str,
        config: SandboxConfig,
    ) -> MicroVMInfo:
        """Create and boot a Firecracker microVM for a session.

        Steps:
          1. Create workspace overlay (writable copy-on-write layer)
          2. Create tap device for networking (if enabled)
          3. Write Firecracker VM config JSON
          4. Launch Firecracker process
          5. Wait for VM to boot (check via API socket)
        """
        vm_id = session_id[:12]
        vm_dir = self.workspaces_root / session_id
        vm_dir.mkdir(parents=True, exist_ok=True)

        log_dir = self.logs_root / session_id
        log_dir.mkdir(parents=True, exist_ok=True)

        socket_path = self.sockets_root / f"{vm_id}.sock"
        vsock_path = self.sockets_root / f"{vm_id}.vsock"
        log_path = log_dir / "vm.log"
        console_log = log_dir / "console.log"

        # 1. Create workspace overlay (copy-on-write on top of rootfs)
        workspace_overlay = vm_dir / "workspace.ext4"
        await self._create_workspace_overlay(
            workspace_overlay, config.disk_size_mb
        )

        # 2. Create tap device for networking
        tap_device = None
        if config.allow_network:
            tap_device = f"{config.tap_device_prefix}{vm_id[:8]}"
            await self._create_tap_device(tap_device)

        # 3. Build Firecracker config
        fc_config = self._build_vm_config(
            vm_id=vm_id,
            config=config,
            workspace_overlay=workspace_overlay,
            vsock_path=vsock_path,
            console_log=console_log,
            tap_device=tap_device,
        )

        config_path = vm_dir / "vm-config.json"
        config_path.write_text(json.dumps(fc_config, indent=2))

        # Write init config (env vars, entrypoint) to a file that the
        # VM's init process will read on boot
        init_config = {
            "env": config.env,
            "entrypoint": config.entrypoint,
            "timeout_seconds": config.timeout_seconds,
        }
        init_config_path = vm_dir / "init-config.json"
        init_config_path.write_text(json.dumps(init_config, indent=2))

        # 4. Launch Firecracker
        if config.use_jailer:
            cmd = self._build_jailer_command(
                vm_id=vm_id,
                config=config,
                config_path=config_path,
                socket_path=socket_path,
            )
        else:
            cmd = [
                self.fc_bin,
                "--api-sock", str(socket_path),
                "--config-file", str(config_path),
                "--log-path", str(log_path),
                "--level", "Warning",
            ]

        logger.info(f"Starting Firecracker VM {vm_id} for session {session_id}")

        log_file = open(log_path, "w")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # 5. Wait for API socket to appear (VM is booting)
        await self._wait_for_socket(socket_path, timeout=10.0)

        info = MicroVMInfo(
            session_id=session_id,
            vm_id=vm_id,
            pid=process.pid,
            api_socket=socket_path,
            vsock_path=vsock_path,
            workspace_path=vm_dir,
            log_path=log_path,
            tap_device=tap_device,
            config=config,
            process=process,
        )
        self._vms[session_id] = info

        logger.info(
            f"VM {vm_id} booted for session {session_id} "
            f"(pid={process.pid}, vcpus={config.vcpu_count}, mem={config.memory_mb}MB)"
        )
        return info

    async def terminate_session(self, session_id: str) -> None:
        """Stop a VM and clean up all resources."""
        info = self._vms.get(session_id)
        if info is None:
            logger.warning(f"Session {session_id} not found")
            return

        # Send shutdown via API (graceful)
        try:
            await self._send_vm_action(info.api_socket, "SendCtrlAltDel")
            # Wait up to 5s for graceful shutdown
            if info.process:
                try:
                    await asyncio.wait_for(info.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            pass

        # Force kill if still running
        if info.process and info.process.returncode is None:
            try:
                os.killpg(os.getpgid(info.pid), signal.SIGKILL)
                await info.process.wait()
            except ProcessLookupError:
                pass

        # Clean up tap device
        if info.tap_device:
            await self._delete_tap_device(info.tap_device)

        # Clean up API socket
        if info.api_socket.exists():
            info.api_socket.unlink()

        del self._vms[session_id]
        logger.info(f"Session {session_id} (VM {info.vm_id}) terminated")

    async def get_session_info(self, session_id: str) -> Optional[MicroVMInfo]:
        """Get info about a running VM."""
        info = self._vms.get(session_id)
        if info is None:
            return None

        # Check if process is still alive
        if info.process and info.process.returncode is not None:
            del self._vms[session_id]
            return None

        return info

    async def get_session_status(self, session_id: str) -> str:
        """Get VM status: running, terminated, unknown."""
        info = await self.get_session_info(session_id)
        if info is None:
            return "terminated"

        # Query Firecracker API for VM state
        try:
            state = await self._get_vm_state(info.api_socket)
            return state  # "Running", "Paused", etc.
        except Exception:
            return "unknown"

    def active_session_count(self) -> int:
        return len(self._vms)

    def active_sessions(self) -> dict[str, MicroVMInfo]:
        return dict(self._vms)

    async def stream_logs(self, session_id: str) -> AsyncIterator[str]:
        """Stream VM console log."""
        info = self._vms.get(session_id)
        if info is None:
            yield "Session not found\n"
            return

        console_log = info.log_path.parent / "console.log"
        if not console_log.exists():
            yield "Console log not found\n"
            return

        with open(console_log, "r") as f:
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    if info.process and info.process.returncode is not None:
                        break
                    await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Firecracker configuration
    # ------------------------------------------------------------------

    def _build_vm_config(
        self,
        vm_id: str,
        config: SandboxConfig,
        workspace_overlay: Path,
        vsock_path: Path,
        console_log: Path,
        tap_device: Optional[str],
    ) -> dict:
        """Build Firecracker VM configuration JSON.

        Reference: https://github.com/firecracker-microvm/firecracker/blob/main/docs/api_requests/README.md
        """
        fc_config = {
            "boot-source": {
                "kernel_image_path": config.kernel_path,
                "boot_args": (
                    "console=ttyS0 reboot=k panic=1 pci=off "
                    "init=/sbin/agent-init "  # Custom init that reads init-config.json
                    "quiet"
                ),
            },
            "drives": [
                {
                    "drive_id": "rootfs",
                    "path_on_host": config.rootfs_path,
                    "is_root_device": True,
                    "is_read_only": True,  # Rootfs is immutable
                },
                {
                    "drive_id": "workspace",
                    "path_on_host": str(workspace_overlay),
                    "is_root_device": False,
                    "is_read_only": False,  # Workspace is writable
                },
            ],
            "machine-config": {
                "vcpu_count": config.vcpu_count,
                "mem_size_mib": config.memory_mb,
                "smt": False,  # No hyperthreading (security)
            },
            "vsock": {
                "guest_cid": 3,  # Standard guest CID
                "uds_path": str(vsock_path),
            },
            "logger": {
                "log_path": str(console_log),
                "level": "Warning",
                "show_level": True,
                "show_log_origin": True,
            },
        }

        # Network interface (only if allowed)
        if tap_device:
            fc_config["network-interfaces"] = [
                {
                    "iface_id": "eth0",
                    "guest_mac": self._generate_mac(vm_id),
                    "host_dev_name": tap_device,
                }
            ]

        return fc_config

    def _build_jailer_command(
        self,
        vm_id: str,
        config: SandboxConfig,
        config_path: Path,
        socket_path: Path,
    ) -> list[str]:
        """Build jailer command for defense-in-depth.

        The jailer runs Firecracker inside a chroot with:
        - Separate UID/GID (no root inside jailer)
        - seccomp filter
        - cgroup isolation
        - chroot filesystem
        """
        return [
            self.jailer_bin,
            "--id", vm_id,
            "--exec-file", self.fc_bin,
            "--uid", str(config.jailer_uid),
            "--gid", str(config.jailer_gid),
            "--chroot-base-dir", str(self.workspaces_root / "jails"),
            "--",
            "--api-sock", str(socket_path),
            "--config-file", str(config_path),
        ]

    # ------------------------------------------------------------------
    # VM communication (via API socket)
    # ------------------------------------------------------------------

    async def _send_vm_action(self, socket_path: Path, action: str) -> None:
        """Send an action to a VM via Firecracker API."""
        cmd = [
            "curl", "--unix-socket", str(socket_path),
            "-X", "PUT",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"action_type": action}),
            "http://localhost/actions",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

    async def _get_vm_state(self, socket_path: Path) -> str:
        """Get VM state via Firecracker API."""
        cmd = [
            "curl", "--unix-socket", str(socket_path),
            "-s", "http://localhost/",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            try:
                data = json.loads(stdout.decode())
                return data.get("state", "unknown")
            except json.JSONDecodeError:
                pass
        return "unknown"

    # ------------------------------------------------------------------
    # Resource helpers
    # ------------------------------------------------------------------

    async def _create_workspace_overlay(self, path: Path, size_mb: int) -> None:
        """Create an empty ext4 filesystem for the workspace drive."""
        proc = await asyncio.create_subprocess_exec(
            "dd", "if=/dev/zero", f"of={path}",
            f"bs=1M", f"count={size_mb}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        proc = await asyncio.create_subprocess_exec(
            "mkfs.ext4", "-F", "-q", str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def _create_tap_device(self, tap_name: str) -> None:
        """Create a TAP device for VM networking."""
        cmds = [
            ["ip", "tuntap", "add", "dev", tap_name, "mode", "tap"],
            ["ip", "link", "set", tap_name, "up"],
        ]
        for cmd in cmds:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()

    async def _delete_tap_device(self, tap_name: str) -> None:
        """Delete a TAP device."""
        proc = await asyncio.create_subprocess_exec(
            "ip", "tuntap", "del", "dev", tap_name, "mode", "tap",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def _wait_for_socket(self, socket_path: Path, timeout: float = 10.0) -> None:
        """Wait for Firecracker API socket to appear."""
        elapsed = 0.0
        interval = 0.1
        while elapsed < timeout:
            if socket_path.exists():
                return
            await asyncio.sleep(interval)
            elapsed += interval
        logger.warning(f"Timeout waiting for API socket {socket_path}")

    @staticmethod
    def _generate_mac(vm_id: str) -> str:
        """Generate a deterministic MAC address from VM ID."""
        # Use first 6 chars of vm_id as hex for last 3 octets
        h = vm_id.ljust(6, "0")[:6]
        return f"02:FC:00:{h[0:2]}:{h[2:4]}:{h[4:6]}"

    async def cleanup_dead(self) -> int:
        """Clean up VMs whose processes have exited."""
        dead = [
            sid for sid, info in self._vms.items()
            if info.process and info.process.returncode is not None
        ]
        for sid in dead:
            info = self._vms[sid]
            if info.tap_device:
                await self._delete_tap_device(info.tap_device)
            if info.api_socket.exists():
                info.api_socket.unlink()
            del self._vms[sid]

        if dead:
            logger.info(f"Cleaned up {len(dead)} dead VMs")
        return len(dead)


__all__ = ["FirecrackerSandbox", "SandboxConfig", "MicroVMInfo"]
