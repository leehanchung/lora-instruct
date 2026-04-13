"""Kubernetes session orchestrator for managed agent jobs.

This is the original orchestrator that maps each session to a K8s Job/Pod.
Simpler to set up but has ~2-5s startup overhead per session. Good for
dev/staging or when you don't need sub-second session startup.
"""

import json
import logging
from typing import AsyncIterator, Optional

from kubernetes_asyncio import client, config, watch
from kubernetes_asyncio.client import ApiException

from .base import BaseOrchestrator, ResourceRequest, SessionInfo, SessionStatus

logger = logging.getLogger(__name__)


class K8sOrchestrator(BaseOrchestrator):
    """Manages agent sessions as Kubernetes Jobs."""

    def __init__(self, namespace: str = "managed-agents") -> None:
        """Initialize async Kubernetes client.

        Args:
            namespace: Kubernetes namespace for session jobs
        """
        self.namespace = namespace
        self.batch_api: Optional[client.BatchV1Api] = None
        self.core_api: Optional[client.CoreV1Api] = None
        self.network_api: Optional[client.NetworkingV1Api] = None

    async def _ensure_client(self) -> None:
        """Ensure Kubernetes client is initialized."""
        if self.batch_api is None:
            try:
                await config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

            self.batch_api = client.BatchV1Api()
            self.core_api = client.CoreV1Api()
            self.network_api = client.NetworkingV1Api()

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
        """Create a Kubernetes Job for a session.

        Args:
            session_id: Unique session identifier
            agent_config: Agent configuration (model, tools, etc.)
            environment_config: Environment configuration (image, packages, resources)
            redis_url: Redis connection URL for state management
            api_base_url: Base URL for API callbacks

        Returns:
            SessionInfo with initial status.

        Raises:
            Exception: If job creation fails
        """
        await self._ensure_client()

        job_name = f"session-{session_id[:12]}"

        # Container image
        image = environment_config.get("image", "managed-agents-runtime:latest")

        # Environment variables
        env_vars = [
            client.V1EnvVar(name="SESSION_ID", value=session_id),
            client.V1EnvVar(
                name="AGENT_CONFIG", value=json.dumps(agent_config)
            ),
            client.V1EnvVar(name="REDIS_URL", value=redis_url),
            client.V1EnvVar(name="API_BASE_URL", value=api_base_url),
            client.V1EnvVar(name="WORKSPACE_DIR", value="/workspace"),
        ]

        # Resource requests/limits
        resources_config = environment_config.get("resources", {})
        cpu_request = resources_config.get("cpu_request", "1")
        cpu_limit = resources_config.get("cpu_limit", "2")
        memory_request = resources_config.get("memory_request", "2Gi")
        memory_limit = resources_config.get("memory_limit", "4Gi")

        resources = client.V1ResourceRequirements(
            requests={"cpu": cpu_request, "memory": memory_request},
            limits={"cpu": cpu_limit, "memory": memory_limit},
        )

        # Volumes for workspace
        volumes = [
            client.V1Volume(
                name="workspace", empty_dir=client.V1EmptyDirVolumeSource()
            ),
            client.V1Volume(
                name="tmp", empty_dir=client.V1EmptyDirVolumeSource()
            ),
        ]

        volume_mounts = [
            client.V1VolumeMount(name="workspace", mount_path="/workspace"),
            client.V1VolumeMount(name="tmp", mount_path="/tmp"),
        ]

        # Main container
        container = client.V1Container(
            name="session-agent",
            image=image,
            image_pull_policy="IfNotPresent",
            env=env_vars,
            resources=resources,
            volume_mounts=volume_mounts,
            security_context=client.V1SecurityContext(
                run_as_non_root=True,
                run_as_user=1000,
                read_only_root_filesystem=True,
                allow_privilege_escalation=False,
                capabilities=client.V1Capabilities(drop=["ALL"]),
            ),
        )

        # Init container for package installation
        init_containers = []
        if "packages" in environment_config:
            init_container = self._build_init_container(
                environment_config["packages"], image
            )
            init_containers.append(init_container)

        # Pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            init_containers=init_containers,
            volumes=volumes,
            restart_policy="Never",
            service_account_name="managed-agents",
            security_context=client.V1PodSecurityContext(
                fs_group=1000,
                run_as_non_root=True,
            ),
        )

        # Job spec
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    name=job_name,
                    labels={"app": "managed-agents", "session_id": session_id},
                ),
                spec=pod_spec,
            ),
            backoff_limit=0,
            ttl_seconds_after_finished=3600,
        )

        # Job metadata
        job_metadata = client.V1ObjectMeta(
            name=job_name,
            namespace=self.namespace,
            labels={"app": "managed-agents", "session_id": session_id},
        )

        # Create Job
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=job_metadata,
            spec=job_spec,
        )

        try:
            created_job = await self.batch_api.create_namespaced_job(
                namespace=self.namespace, body=job
            )
            logger.info(f"Created job {job_name} for session {session_id}")
            return SessionInfo(
                session_id=session_id,
                status=SessionStatus.PENDING,
                metadata={"job_name": created_job.metadata.name},
            )
        except ApiException as e:
            logger.error(
                f"Failed to create job {job_name}: {e.status} {e.reason}"
            )
            raise

    async def terminate_session(self, session_id: str) -> None:
        """Delete a session Job.

        Args:
            session_id: Session identifier

        Raises:
            Exception: If deletion fails
        """
        await self._ensure_client()

        job_name = f"session-{session_id[:12]}"

        try:
            await self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Background",
            )
            logger.info(f"Deleted job {job_name} for session {session_id}")
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Job {job_name} not found")
            else:
                logger.error(
                    f"Failed to delete job {job_name}: {e.status} {e.reason}"
                )
                raise

    async def get_session_status(self, session_id: str) -> SessionInfo:
        """Get session Job status.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo with current status.
        """
        await self._ensure_client()

        job_name = f"session-{session_id[:12]}"

        try:
            job = await self.batch_api.read_namespaced_job(
                name=job_name, namespace=self.namespace
            )

            # Map Job status to session status
            if job.status.succeeded:
                status = SessionStatus.TERMINATED
            elif job.status.failed:
                status = SessionStatus.TERMINATED
            elif job.status.active:
                status = SessionStatus.RUNNING
            elif job.status.conditions:
                status = SessionStatus.PENDING
                for condition in job.status.conditions:
                    if condition.type == "Suspended":
                        status = SessionStatus.IDLE
            else:
                status = SessionStatus.PENDING

            return SessionInfo(
                session_id=session_id,
                status=status,
                metadata={"job_name": job_name},
            )

        except ApiException as e:
            if e.status == 404:
                return SessionInfo(
                    session_id=session_id, status=SessionStatus.TERMINATED
                )
            logger.error(
                f"Failed to get status for {job_name}: {e.status} {e.reason}"
            )
            raise

    async def stream_logs(
        self, session_id: str, container_name: str = "session-agent"
    ) -> AsyncIterator[str]:
        """Stream pod logs for debugging.

        Args:
            session_id: Session identifier
            container_name: Container name to stream logs from

        Yields:
            Log lines
        """
        await self._ensure_client()

        job_name = f"session-{session_id[:12]}"

        try:
            # Get pod for the job
            pods = await self.core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}",
            )

            if not pods.items:
                logger.warning(f"No pods found for job {job_name}")
                return

            pod_name = pods.items[0].metadata.name

            # Stream logs
            log_stream = watch.Watch()
            async for event in log_stream.stream(
                self.core_api.read_namespaced_pod_log,
                name=pod_name,
                namespace=self.namespace,
                container=container_name,
                follow=True,
                _preload_content=False,
            ):
                yield event.decode("utf-8")

        except ApiException as e:
            logger.error(
                f"Failed to stream logs for {job_name}: {e.status} {e.reason}"
            )
            raise

    def _build_init_container(
        self, packages_config: dict, image: str
    ) -> client.V1Container:
        """Build init container for package installation.

        Args:
            packages_config: Package configuration dict
            image: Base image to use

        Returns:
            V1Container for init container
        """
        commands = []

        # Handle apt packages (must come first)
        if "apt" in packages_config:
            apt_packages = " ".join(packages_config["apt"])
            commands.append("apt-get update")
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

        # Handle cargo packages
        if "cargo" in packages_config:
            for pkg in packages_config["cargo"]:
                commands.append(f"cargo install {pkg}")

        # Handle go packages
        if "go" in packages_config:
            for pkg in packages_config["go"]:
                commands.append(f"go install {pkg}")

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


__all__ = ["K8sOrchestrator"]
