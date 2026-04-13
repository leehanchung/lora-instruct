"""Rootfs builder: converts Docker images into Firecracker-compatible ext4 rootfs.

Firecracker boots from raw ext4 filesystem images, not OCI containers.
This tool bridges the gap:

  Dockerfile → docker build → OCI image → export → ext4 rootfs

Usage:
  # Build a single harness rootfs
  python -m src.orchestrator.rootfs_builder --harness claude-code

  # Build all harness rootfs images
  python -m src.orchestrator.rootfs_builder --all

  # Build from a custom Dockerfile
  python -m src.orchestrator.rootfs_builder --dockerfile path/to/Dockerfile --output my-agent.ext4

The resulting .ext4 files go into /var/lib/managed-agents/rootfs/ and are
used by FirecrackerSandbox as read-only root drives.

The builder also injects the agent-init binary into /sbin/agent-init,
which is the first process that runs inside the VM. It reads
/etc/init-config.json (written by the host at boot) and launches the
harness entrypoint with the right env vars.
"""

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .harness import BUILTIN_HARNESSES, HarnessDefinition

logger = logging.getLogger(__name__)

DEFAULT_ROOTFS_DIR = os.getenv(
    "ROOTFS_ROOT", "/var/lib/managed-agents/rootfs"
)
DEFAULT_INIT_PATH = os.getenv(
    "AGENT_INIT_PATH", ""  # If empty, we generate a shell-script init
)

# Mapping from harness name → Dockerfile path (relative to infra/)
HARNESS_DOCKERFILES: dict[str, str] = {
    "claude-code": "harnesses/claude-code.Dockerfile",
    "custom-python": "harnesses/custom-python.Dockerfile",
    "swe-agent": "harnesses/swe-agent.Dockerfile",
    "aider": "harnesses/aider.Dockerfile",
    # openhands: uses upstream image, no local Dockerfile
}


def _generate_shell_init() -> str:
    """Generate a shell-script init for the VM.

    This runs as PID 1 inside the Firecracker VM:
      1. Mounts /proc, /sys, /dev, /tmp
      2. Mounts the workspace drive (/dev/vdb → /workspace)
      3. Reads /etc/init-config.json for env vars and entrypoint
      4. Exports env vars
      5. Execs the harness entrypoint
    """
    return r"""#!/bin/busybox sh
# agent-init: PID 1 inside Firecracker microVM
# Reads /etc/init-config.json and launches the harness entrypoint

set -e

# Mount essential filesystems
/bin/busybox mount -t proc proc /proc
/bin/busybox mount -t sysfs sys /sys
/bin/busybox mount -t devtmpfs dev /dev
/bin/busybox mount -t tmpfs tmpfs /tmp
/bin/busybox mount -t tmpfs tmpfs /run

# Mount workspace drive (second virtio block device)
if [ -b /dev/vdb ]; then
    /bin/busybox mkdir -p /workspace
    /bin/busybox mount /dev/vdb /workspace
fi

# Set hostname
/bin/busybox hostname agent

# Configure network if eth0 exists
if [ -d /sys/class/net/eth0 ]; then
    /bin/busybox ip link set eth0 up
    /bin/busybox ip addr add 172.16.0.2/24 dev eth0
    /bin/busybox ip route add default via 172.16.0.1
    # DNS
    echo "nameserver 8.8.8.8" > /etc/resolv.conf
    echo "nameserver 8.8.4.4" >> /etc/resolv.conf
fi

# Read init config (env vars, entrypoint, timeout)
INIT_CONFIG="/etc/init-config.json"
if [ ! -f "$INIT_CONFIG" ]; then
    echo "FATAL: $INIT_CONFIG not found"
    /bin/busybox poweroff -f
fi

# Parse JSON config using a minimal approach (busybox-compatible)
# The host writes this file before VM boot
ENTRYPOINT=""
TIMEOUT=0

# Extract env vars and export them
# Format in JSON: {"env": {"KEY": "VALUE", ...}, "entrypoint": [...], "timeout_seconds": N}
# We use a simple line-by-line parser since busybox may not have jq

if command -v jq > /dev/null 2>&1; then
    # If jq is available, use it
    for key in $(jq -r '.env // {} | keys[]' "$INIT_CONFIG" 2>/dev/null); do
        val=$(jq -r ".env[\"$key\"]" "$INIT_CONFIG")
        export "$key=$val"
    done
    ENTRYPOINT=$(jq -r '.entrypoint // [] | join(" ")' "$INIT_CONFIG" 2>/dev/null)
    TIMEOUT=$(jq -r '.timeout_seconds // 0' "$INIT_CONFIG" 2>/dev/null)
elif command -v python3 > /dev/null 2>&1; then
    # Fall back to Python for JSON parsing
    eval "$(python3 -c "
import json, sys
with open('$INIT_CONFIG') as f:
    cfg = json.load(f)
for k, v in cfg.get('env', {}).items():
    print(f'export {k}=\"{v}\"')
ep = cfg.get('entrypoint', [])
if ep:
    print(f'ENTRYPOINT=\"{\" \".join(ep)}\"')
print(f'TIMEOUT={cfg.get(\"timeout_seconds\", 0)}')
")"
fi

# Switch to non-root user if available
AGENT_UID=1000
if id -u agent > /dev/null 2>&1; then
    RUN_AS="su -s /bin/sh agent -c"
else
    RUN_AS="sh -c"
fi

echo "agent-init: starting harness: $ENTRYPOINT"
echo "agent-init: timeout: ${TIMEOUT}s (0=none)"

# Launch the entrypoint
if [ "$TIMEOUT" -gt 0 ] 2>/dev/null; then
    # Timeout: kill after N seconds
    (
        sleep "$TIMEOUT"
        echo "agent-init: timeout reached ($TIMEOUT s), shutting down"
        kill -TERM 1
    ) &
fi

cd /workspace

if [ -n "$ENTRYPOINT" ]; then
    exec $RUN_AS "$ENTRYPOINT"
else
    echo "agent-init: no entrypoint configured, dropping to shell"
    exec /bin/sh
fi
"""


def build_rootfs_from_docker(
    image_tag: str,
    output_path: str,
    size_mb: int = 2048,
    init_script: str | None = None,
) -> None:
    """Build an ext4 rootfs from a Docker image.

    Steps:
      1. Create a temporary container from the image (no start)
      2. Export the container filesystem to a tar
      3. Create an empty ext4 image of the desired size
      4. Mount it and extract the tar into it
      5. Inject agent-init into /sbin/agent-init
      6. Clean up
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rootfs-build-") as tmpdir:
        tmp = Path(tmpdir)
        tar_path = tmp / "rootfs.tar"
        mount_point = tmp / "mnt"
        mount_point.mkdir()

        logger.info(f"Creating container from {image_tag}...")
        container_id = subprocess.check_output(
            ["docker", "create", image_tag, "/bin/true"],
            text=True,
        ).strip()

        try:
            # Export container filesystem
            logger.info(f"Exporting container {container_id[:12]} filesystem...")
            with open(tar_path, "wb") as tar_file:
                subprocess.run(
                    ["docker", "export", container_id],
                    stdout=tar_file,
                    check=True,
                )

            # Create ext4 image
            logger.info(f"Creating {size_mb}MB ext4 image at {output}...")
            subprocess.run(
                ["dd", "if=/dev/zero", f"of={output}", "bs=1M", f"count={size_mb}"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["mkfs.ext4", "-F", "-q", str(output)],
                check=True,
                capture_output=True,
            )

            # Mount and populate
            logger.info("Mounting and extracting filesystem...")
            subprocess.run(
                ["mount", "-o", "loop", str(output), str(mount_point)],
                check=True,
            )

            try:
                # Extract tar into mounted filesystem
                subprocess.run(
                    ["tar", "xf", str(tar_path), "-C", str(mount_point)],
                    check=True,
                )

                # Ensure essential directories exist
                for d in ["proc", "sys", "dev", "tmp", "run", "workspace", "etc"]:
                    (mount_point / d).mkdir(exist_ok=True)

                # Inject agent-init
                init_path = mount_point / "sbin" / "agent-init"
                init_path.parent.mkdir(parents=True, exist_ok=True)

                if init_script:
                    init_content = init_script
                else:
                    init_content = _generate_shell_init()

                init_path.write_text(init_content)
                os.chmod(init_path, 0o755)

                # Ensure busybox is available (for the shell init)
                busybox = mount_point / "bin" / "busybox"
                if not busybox.exists():
                    # Try to copy busybox from host
                    host_busybox = shutil.which("busybox")
                    if host_busybox:
                        shutil.copy2(host_busybox, busybox)
                        os.chmod(busybox, 0o755)
                    else:
                        logger.warning(
                            "busybox not found on host — init may fail if "
                            "the image doesn't have coreutils"
                        )

                logger.info("Rootfs populated successfully")
            finally:
                subprocess.run(["umount", str(mount_point)], check=True)

        finally:
            subprocess.run(
                ["docker", "rm", container_id],
                capture_output=True,
            )

    logger.info(f"Rootfs built: {output} ({size_mb}MB)")


def build_harness_rootfs(
    harness_name: str,
    output_dir: str = DEFAULT_ROOTFS_DIR,
    dockerfile_dir: str = ".",
    size_mb: int = 2048,
) -> str:
    """Build rootfs for a named built-in harness.

    Returns path to the built rootfs image.
    """
    if harness_name not in BUILTIN_HARNESSES:
        raise ValueError(
            f"Unknown harness: {harness_name}. "
            f"Available: {list(BUILTIN_HARNESSES.keys())}"
        )

    harness = BUILTIN_HARNESSES[harness_name]
    output_path = os.path.join(output_dir, f"{harness_name}.ext4")

    # Check if we have a Dockerfile for this harness
    dockerfile = HARNESS_DOCKERFILES.get(harness_name)
    if dockerfile:
        dockerfile_path = os.path.join(dockerfile_dir, dockerfile)
        if not os.path.exists(dockerfile_path):
            raise FileNotFoundError(
                f"Dockerfile not found: {dockerfile_path}"
            )

        # Build Docker image first
        logger.info(f"Building Docker image for {harness_name}...")
        subprocess.run(
            [
                "docker", "build",
                "-f", dockerfile_path,
                "-t", harness.image,
                dockerfile_dir,
            ],
            check=True,
        )

    elif not harness.image:
        raise ValueError(
            f"Harness '{harness_name}' has no Dockerfile and no image specified"
        )

    # Convert Docker image to rootfs
    build_rootfs_from_docker(
        image_tag=harness.image,
        output_path=output_path,
        size_mb=size_mb,
    )

    return output_path


def build_all_harnesses(
    output_dir: str = DEFAULT_ROOTFS_DIR,
    dockerfile_dir: str = ".",
) -> dict[str, str]:
    """Build rootfs for all harnesses that have Dockerfiles.

    Returns dict of harness_name → rootfs_path.
    """
    results = {}
    for name in HARNESS_DOCKERFILES:
        try:
            path = build_harness_rootfs(
                harness_name=name,
                output_dir=output_dir,
                dockerfile_dir=dockerfile_dir,
            )
            results[name] = path
            logger.info(f"✓ {name}: {path}")
        except Exception as e:
            logger.error(f"✗ {name}: {e}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Firecracker rootfs images from harness Dockerfiles"
    )
    parser.add_argument(
        "--harness", "-n",
        help="Build rootfs for a specific harness name",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Build rootfs for all harnesses with Dockerfiles",
    )
    parser.add_argument(
        "--dockerfile", "-f",
        help="Build rootfs from a custom Dockerfile",
    )
    parser.add_argument(
        "--image", "-i",
        help="Build rootfs from an existing Docker image tag",
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Output path for rootfs image",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_ROOTFS_DIR,
        help=f"Output directory (default: {DEFAULT_ROOTFS_DIR})",
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=2048,
        help="Rootfs size in MB (default: 2048)",
    )
    parser.add_argument(
        "--dockerfile-dir",
        default=".",
        help="Directory context for Docker builds (default: .)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available harnesses and exit",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.list:
        print("Built-in harnesses:")
        for name, h in BUILTIN_HARNESSES.items():
            df = HARNESS_DOCKERFILES.get(name, "(no Dockerfile)")
            print(f"  {name:25s} {h.image:45s} {df}")
        return

    if args.all:
        results = build_all_harnesses(
            output_dir=args.output_dir,
            dockerfile_dir=args.dockerfile_dir,
        )
        print(f"\nBuilt {len(results)} rootfs images:")
        for name, path in results.items():
            print(f"  {name}: {path}")
        return

    if args.harness:
        path = build_harness_rootfs(
            harness_name=args.harness,
            output_dir=args.output_dir,
            dockerfile_dir=args.dockerfile_dir,
            size_mb=args.size,
        )
        print(f"Built rootfs: {path}")
        return

    if args.dockerfile:
        # Build Docker image from Dockerfile, then convert
        tag = f"rootfs-build-custom:{args.output or 'latest'}"
        subprocess.run(
            ["docker", "build", "-f", args.dockerfile, "-t", tag, args.dockerfile_dir],
            check=True,
        )
        output = args.output or os.path.join(args.output_dir, "custom.ext4")
        build_rootfs_from_docker(tag, output, size_mb=args.size)
        print(f"Built rootfs: {output}")
        return

    if args.image:
        output = args.output or os.path.join(
            args.output_dir, f"{args.image.split(':')[0].split('/')[-1]}.ext4"
        )
        build_rootfs_from_docker(args.image, output, size_mb=args.size)
        print(f"Built rootfs: {output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
