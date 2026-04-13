"""Harness definitions: what runs inside each Firecracker microVM sandbox.

A Harness is a rootfs image + entrypoint + config that defines an agent
runtime. The managed-agents platform is harness-agnostic — it provides
isolation (via Firecracker microVMs), scheduling, and I/O routing, and
the harness provides the agent logic.

Examples:
  - Claude Code + Discord: the `claude` CLI with MCP plugins
  - Custom Python agent: a custom agent_loop.py with tool implementations
  - SWE-agent: Princeton's SWE-agent for automated issue resolution
  - OpenHands (Devin-like): CodeAct agent with browser + shell
  - Aider: AI pair programming in terminal
  - Any arbitrary agent: bring your own rootfs image

Each harness is an ext4 rootfs image. The platform:
  1. Resolves the harness rootfs (pre-built or built from Dockerfile)
  2. Creates a Firecracker microVM with the rootfs + writable workspace
  3. Boots the VM with its own kernel, network, and filesystem
  4. Routes I/O (Discord, Slack, API, etc.) via env vars or MCP
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SandboxRuntime(str, Enum):
    """Sandbox technology that isolates the harness.

    Isolation strength (weakest → strongest):
      FIRECRACKER:   Firecracker microVM. Own kernel, ~125ms boot, ~5MB.
                     Gold standard — every session gets its own kernel.
      GVISOR:        gVisor (runsc). User-space kernel intercepts syscalls.
                     ~200ms start. No dedicated kernel, but strong.
      RUNC:          Standard Linux containers. cgroups + namespaces.
                     ~100ms start. Shared kernel. Only for trusted code.
    """

    FIRECRACKER = "firecracker"  # Default — full microVM isolation
    GVISOR = "gvisor"  # gVisor — syscall interposition
    RUNC = "runc"  # Standard container (not recommended)


# Backward compat alias
ContainerRuntime = SandboxRuntime


@dataclass
class HarnessNetworkPolicy:
    """Network policy for the sandbox.

    Default: deny all. Explicitly allow what's needed.
    For Firecracker: implemented via iptables rules on the host TAP device.
    For containers: implemented via --network=none or CNI policies.
    """

    enabled: bool = False  # False = no network at all
    allowed_domains: list[str] = field(default_factory=list)
    # Common presets that expand into domain lists
    allow_anthropic_api: bool = False  # api.anthropic.com
    allow_discord: bool = False  # discord.com, gateway.discord.gg, cdn.discordapp.com
    allow_github: bool = False  # github.com, api.github.com
    allow_pypi: bool = False  # pypi.org, files.pythonhosted.org
    allow_npm: bool = False  # registry.npmjs.org

    def resolved_domains(self) -> list[str]:
        """Expand presets into concrete domain list."""
        domains = list(self.allowed_domains)
        if self.allow_anthropic_api:
            domains += ["api.anthropic.com"]
        if self.allow_discord:
            domains += ["discord.com", "gateway.discord.gg", "cdn.discordapp.com"]
        if self.allow_github:
            domains += ["github.com", "api.github.com", "raw.githubusercontent.com"]
        if self.allow_pypi:
            domains += ["pypi.org", "files.pythonhosted.org"]
        if self.allow_npm:
            domains += ["registry.npmjs.org"]
        return list(set(domains))


@dataclass
class HarnessDefinition:
    """A harness: rootfs image + config that defines an agent runtime.

    This is the unit of "what kind of agent runs in the sandbox."

    For Firecracker sandboxes:
      - rootfs_path points to a pre-built ext4 filesystem image
      - entrypoint is passed via init-config.json to the VM's init process
      - resources map to VM vcpu/memory allocation

    For backward-compat container sandboxes (runc/gvisor):
      - image is the OCI image reference
      - entrypoint/command override the image defaults
    """

    # Identity
    name: str  # e.g. "claude-code", "swe-agent", "custom-python"
    version: str = "latest"
    description: str = ""

    # Rootfs image (Firecracker) — path to ext4 rootfs on worker
    rootfs_path: str = ""  # e.g. "/var/lib/managed-agents/rootfs/claude-code.ext4"

    # OCI image (container fallback) — used to build rootfs or run directly
    image: str = ""  # e.g. "managed-agents-claude-code:latest"

    # Entrypoint and command (applied inside the sandbox)
    entrypoint: list[str] = field(default_factory=list)  # Override rootfs/image entrypoint
    command: list[str] = field(default_factory=list)  # Arguments to entrypoint

    # Sandbox runtime (isolation level)
    runtime: SandboxRuntime = SandboxRuntime.FIRECRACKER

    # Environment variables baked into the harness definition
    # (session-specific env like API keys are added at launch time)
    base_env: dict[str, str] = field(default_factory=dict)

    # Network
    network: HarnessNetworkPolicy = field(default_factory=HarnessNetworkPolicy)

    # Resource defaults (can be overridden per-session)
    default_vcpus: int = 2  # Firecracker vCPUs
    default_memory_mb: int = 2048  # VM memory
    default_disk_mb: int = 4096  # Workspace overlay size
    default_gpu_count: int = 0  # GPU passthrough (future)

    # Backward compat: cpu_millicores → vcpus conversion
    @property
    def default_cpu_millicores(self) -> int:
        return self.default_vcpus * 1000

    # Security
    read_only_rootfs: bool = True  # Rootfs is always immutable (Firecracker drives)
    use_jailer: bool = True  # Jailer for defense in depth

    # Lifecycle
    default_timeout_seconds: int = 3600  # 0 = no timeout
    health_check_interval_seconds: int = 30

    # Labels for scheduler matching
    # Workers must have these labels to run this harness
    required_worker_labels: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in harness definitions
# ---------------------------------------------------------------------------

HARNESS_CLAUDE_CODE = HarnessDefinition(
    name="claude-code",
    description="Claude Code CLI with MCP plugin support (Discord, Slack, etc.)",
    rootfs_path="/var/lib/managed-agents/rootfs/claude-code.ext4",
    image="managed-agents-claude-code:latest",  # Source for rootfs build
    entrypoint=["claude"],
    base_env={
        "TERM": "xterm-256color",
        "NODE_ENV": "production",
    },
    network=HarnessNetworkPolicy(
        enabled=True,
        allow_anthropic_api=True,
    ),
    default_vcpus=2,
    default_memory_mb=2048,
    default_timeout_seconds=0,  # Long-lived bot
)

HARNESS_CLAUDE_CODE_DISCORD = HarnessDefinition(
    name="claude-code-discord",
    description="Claude Code with Discord MCP plugin pre-configured",
    rootfs_path="/var/lib/managed-agents/rootfs/claude-code.ext4",
    image="managed-agents-claude-code:latest",
    entrypoint=["claude", "--channels", "plugin:discord@claude-plugins-official"],
    base_env={
        "TERM": "xterm-256color",
        "NODE_ENV": "production",
    },
    network=HarnessNetworkPolicy(
        enabled=True,
        allow_anthropic_api=True,
        allow_discord=True,
    ),
    default_vcpus=2,
    default_memory_mb=2048,
    default_timeout_seconds=0,
)

HARNESS_CUSTOM_PYTHON = HarnessDefinition(
    name="custom-python",
    description="Custom Python agent loop with built-in tool implementations",
    rootfs_path="/var/lib/managed-agents/rootfs/custom-python.ext4",
    image="managed-agents-runtime:latest",
    entrypoint=["python3", "-m", "src.runtime.agent_loop"],
    base_env={
        "PYTHONUNBUFFERED": "1",
    },
    network=HarnessNetworkPolicy(
        enabled=True,
        allow_anthropic_api=True,
    ),
    default_vcpus=2,
    default_memory_mb=2048,
    default_timeout_seconds=3600,
)

HARNESS_SWE_AGENT = HarnessDefinition(
    name="swe-agent",
    description="SWE-agent: autonomous software engineering agent (Princeton)",
    rootfs_path="/var/lib/managed-agents/rootfs/swe-agent.ext4",
    image="managed-agents-swe-agent:latest",
    entrypoint=["python", "-m", "swe_agent.run"],
    base_env={
        "SWE_AGENT_CONFIG": "/workspace/.swe-agent.yaml",
    },
    network=HarnessNetworkPolicy(
        enabled=True,
        allow_anthropic_api=True,
        allow_github=True,
    ),
    runtime=SandboxRuntime.FIRECRACKER,
    default_vcpus=2,
    default_memory_mb=4096,
    default_disk_mb=8192,
    default_timeout_seconds=1800,
)

HARNESS_OPENHANDS = HarnessDefinition(
    name="openhands",
    description="OpenHands (formerly OpenDevin): CodeAct agent with browser + shell",
    rootfs_path="/var/lib/managed-agents/rootfs/openhands.ext4",
    image="managed-agents-openhands:latest",
    entrypoint=["python", "-m", "openhands.core.main"],
    base_env={
        "SANDBOX_TYPE": "local",
    },
    network=HarnessNetworkPolicy(
        enabled=True,
        allow_anthropic_api=True,
        allow_github=True,
        allow_pypi=True,
        allow_npm=True,
    ),
    runtime=SandboxRuntime.FIRECRACKER,
    default_vcpus=4,
    default_memory_mb=8192,
    default_disk_mb=16384,
    default_timeout_seconds=3600,
)

HARNESS_AIDER = HarnessDefinition(
    name="aider",
    description="Aider: AI pair programming in terminal",
    rootfs_path="/var/lib/managed-agents/rootfs/aider.ext4",
    image="managed-agents-aider:latest",
    entrypoint=["aider"],
    base_env={
        "AIDER_MODEL": "claude-3-5-sonnet-20241022",
    },
    network=HarnessNetworkPolicy(
        enabled=True,
        allow_anthropic_api=True,
        allow_github=True,
    ),
    default_vcpus=1,
    default_memory_mb=1024,
    default_disk_mb=2048,
    default_timeout_seconds=3600,
)

# Registry of built-in harnesses
BUILTIN_HARNESSES: dict[str, HarnessDefinition] = {
    h.name: h
    for h in [
        HARNESS_CLAUDE_CODE,
        HARNESS_CLAUDE_CODE_DISCORD,
        HARNESS_CUSTOM_PYTHON,
        HARNESS_SWE_AGENT,
        HARNESS_OPENHANDS,
        HARNESS_AIDER,
    ]
}


__all__ = [
    "SandboxRuntime",
    "ContainerRuntime",  # backward compat alias
    "HarnessDefinition",
    "HarnessNetworkPolicy",
    "BUILTIN_HARNESSES",
    "HARNESS_CLAUDE_CODE",
    "HARNESS_CLAUDE_CODE_DISCORD",
    "HARNESS_CUSTOM_PYTHON",
    "HARNESS_SWE_AGENT",
    "HARNESS_OPENHANDS",
    "HARNESS_AIDER",
]
