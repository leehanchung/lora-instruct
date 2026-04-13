"""Container and environment builders for Kubernetes."""

from typing import Any, Optional

from kubernetes_asyncio import client


def build_init_container(packages_config: dict, image: str) -> client.V1Container:
    """Generate an init container for package installation.

    Builds shell commands for pip, npm, and apt-get installation.

    Args:
        packages_config: Dict with keys 'pip', 'npm', 'apt' containing lists of packages
        image: Base container image to use

    Returns:
        V1Container configured as init container
    """
    commands = []

    # Handle apt packages (must come first to update)
    if "apt" in packages_config:
        apt_packages = " ".join(packages_config["apt"])
        commands.insert(0, "apt-get update")
        commands.append(f"apt-get install -y --no-install-recommends {apt_packages}")
        commands.append("apt-get clean && rm -rf /var/lib/apt/lists/*")

    # Handle pip packages
    if "pip" in packages_config:
        pip_packages = " ".join(packages_config["pip"])
        commands.append(f"pip install --no-cache-dir {pip_packages}")

    # Handle npm packages
    if "npm" in packages_config:
        npm_packages = " ".join(packages_config["npm"])
        commands.append(f"npm install --global --no-save {npm_packages}")

    combined_command = " && ".join(commands)

    return client.V1Container(
        name="install-packages",
        image=image,
        command=["sh", "-c"],
        args=[combined_command],
        security_context=client.V1SecurityContext(
            run_as_user=0,  # Root for installation
        ),
    )


def build_network_policy(
    session_id: str, networking_config: Optional[dict] = None
) -> client.V1NetworkPolicy:
    """Create a NetworkPolicy for limited networking.

    Restricts egress to specified hosts only.

    Args:
        session_id: Session identifier for policy naming
        networking_config: Dict with 'allowed_hosts' list

    Returns:
        V1NetworkPolicy configured with egress rules
    """
    if networking_config is None:
        networking_config = {}

    allowed_hosts = networking_config.get("allowed_hosts", [])
    policy_name = f"session-{session_id[:12]}-network-policy"

    # Build egress rules for allowed hosts
    egress_rules = []

    # Always allow DNS (port 53)
    egress_rules.append(
        client.V1NetworkPolicyEgressRule(
            to=[
                client.V1NetworkPolicyPeer(
                    namespace_selector=client.V1LabelSelector(
                        match_labels={"name": "kube-system"}
                    )
                )
            ],
            ports=[client.V1NetworkPolicyPort(protocol="UDP", port=53)],
        )
    )

    # Allow traffic to specified hosts
    for host in allowed_hosts:
        egress_rules.append(
            client.V1NetworkPolicyEgressRule(
                to=[client.V1NetworkPolicyPeer(ip_block=client.V1IPBlock(cidr=host))]
            )
        )

    # Deny all by default
    policy_spec = client.V1NetworkPolicySpec(
        pod_selector=client.V1LabelSelector(
            match_labels={"session_id": session_id}
        ),
        policy_types=["Ingress", "Egress"],
        ingress=[],  # No ingress
        egress=egress_rules,
    )

    return client.V1NetworkPolicy(
        api_version="networking.k8s.io/v1",
        kind="NetworkPolicy",
        metadata=client.V1ObjectMeta(name=policy_name),
        spec=policy_spec,
    )


def build_resource_mounts(
    resources: Optional[dict],
) -> tuple[list[client.V1Volume], list[client.V1VolumeMount]]:
    """Build volume mounts for resources (GitHub repos, files, etc.).

    Args:
        resources: Dict with 'repos' and 'files' configurations

    Returns:
        Tuple of (volumes, volume_mounts) lists
    """
    volumes: list[client.V1Volume] = []
    volume_mounts: list[client.V1VolumeMount] = []

    if resources is None:
        return volumes, volume_mounts

    # Handle GitHub repositories
    repos = resources.get("repos", [])
    for idx, repo in enumerate(repos):
        mount_path = repo.get("mount_path", f"/repos/{idx}")

        volume_name = f"repo-{idx}"
        volumes.append(
            client.V1Volume(
                name=volume_name,
                empty_dir=client.V1EmptyDirVolumeSource(),
            )
        )
        volume_mounts.append(
            client.V1VolumeMount(
                name=volume_name,
                mount_path=mount_path,
            )
        )

    # Handle mounted files/config maps
    files = resources.get("files", [])
    for idx, file_cfg in enumerate(files):
        name = file_cfg.get("name", f"file-{idx}")
        mount_path = file_cfg.get("mount_path")

        volume_name = f"file-{idx}"
        volumes.append(
            client.V1Volume(
                name=volume_name,
                config_map=client.V1ConfigMapVolumeSource(
                    name=name,
                ),
            )
        )
        volume_mounts.append(
            client.V1VolumeMount(
                name=volume_name,
                mount_path=mount_path,
            )
        )

    return volumes, volume_mounts


__all__ = [
    "build_init_container",
    "build_network_policy",
    "build_resource_mounts",
]
