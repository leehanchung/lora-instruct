"""Multi-session launcher: spin up N Claude Code + Discord instances.

Reads a YAML config defining multiple bot sessions, prepares their
filesystems, and dispatches them to the orchestrator (which places
them on workers via the scheduler).

Usage:
  python -m src.runtime.launcher --config sessions.yaml

Or programmatically:
  launcher = SessionLauncher(orchestrator, config_path="sessions.yaml")
  await launcher.launch_all()
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

from src.orchestrator import (
    BaseOrchestrator,
    ResourceRequest,
    create_orchestrator,
)
from src.runtime.claude_code import (
    ClaudeCodeSessionConfig,
    build_claude_code_command,
    build_multi_session_configs,
    build_session_environment,
    prepare_session_filesystem,
)

logger = logging.getLogger(__name__)


class SessionLauncher:
    """Launches and manages multiple Claude Code + Discord sessions."""

    def __init__(
        self,
        orchestrator: BaseOrchestrator,
        workspaces_root: str = "/var/lib/managed-agents/workspaces",
    ) -> None:
        self.orchestrator = orchestrator
        self.workspaces_root = Path(workspaces_root)
        self._active_sessions: dict[str, ClaudeCodeSessionConfig] = {}

    async def launch_all(
        self, configs: list[ClaudeCodeSessionConfig]
    ) -> list[str]:
        """Launch all sessions concurrently.

        Returns list of session IDs that were successfully started.
        """
        logger.info(f"Launching {len(configs)} Claude Code sessions...")

        tasks = [self._launch_one(config) for config in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        launched = []
        for config, result in zip(configs, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to launch {config.name}: {result}")
            else:
                launched.append(config.session_id)
                logger.info(f"Launched {config.name} ({config.session_id})")

        logger.info(f"Successfully launched {len(launched)}/{len(configs)} sessions")
        return launched

    async def _launch_one(self, config: ClaudeCodeSessionConfig) -> str:
        """Launch a single Claude Code + Discord session."""
        # Prepare the filesystem (config files, access.json, .env, etc.)
        session_dir = prepare_session_filesystem(config, self.workspaces_root)

        # Build the environment config for the orchestrator
        claude_cmd = build_claude_code_command(config)
        claude_env = build_session_environment(config, state_dir="/home/agent/.claude")

        environment_config = {
            # The command to run inside the sandbox
            "entrypoint": claude_cmd,
            "env": claude_env,

            # Resource requirements
            "resources": {
                "cpu_request": f"{config.cpu_millicores}m",
                "memory_request": f"{config.memory_mb}Mi",
                "gpu_count": config.gpu_count,
                "timeout_seconds": config.timeout_seconds,
            },

            # Network: must allow Anthropic API + Discord API
            "networking": {
                "enabled": True,
                "allowed_hosts": [
                    "api.anthropic.com",
                    "discord.com",
                    "gateway.discord.gg",
                    "cdn.discordapp.com",
                    # For plugin installation
                    "registry.npmjs.org",
                    "github.com",
                ],
            },

            # Scheduling hints
            "scheduling": {
                # Long-lived bots benefit from spread (fault tolerance)
                "anti_affinity_tags": [f"discord-bot:{config.name}"],
                # If multiple bots use the same model, co-locate for cache
                "affinity_tags": [f"model:{config.model}"],
            },
        }

        # Build resource request for the scheduler
        resource_request = ResourceRequest(
            cpu_millicores=config.cpu_millicores,
            memory_mb=config.memory_mb,
            gpu_count=config.gpu_count,
            timeout_seconds=config.timeout_seconds or 0,
            anti_affinity_tags=[f"discord-bot:{config.name}"],
            affinity_tags=[f"model:{config.model}"],
        )

        # Agent config (passed to the sandbox as metadata)
        agent_config = {
            "name": config.name,
            "model": config.model,
            "system_prompt": config.system_prompt[:200] + "..." if len(config.system_prompt) > 200 else config.system_prompt,
            "plugins": config.plugins,
            "type": "claude-code-discord",
        }

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

        # Dispatch to orchestrator
        session_info = await self.orchestrator.create_session(
            session_id=config.session_id,
            agent_config=agent_config,
            environment_config=environment_config,
            redis_url=redis_url,
            api_base_url=api_base_url,
            resources=resource_request,
            tags=[f"discord-bot:{config.name}", f"model:{config.model}"],
        )

        self._active_sessions[config.session_id] = config
        return session_info.session_id

    async def terminate_all(self) -> None:
        """Terminate all active sessions."""
        logger.info(f"Terminating {len(self._active_sessions)} sessions...")
        tasks = [
            self.orchestrator.terminate_session(sid)
            for sid in self._active_sessions
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._active_sessions.clear()

    async def terminate_one(self, session_id: str) -> None:
        """Terminate a specific session."""
        await self.orchestrator.terminate_session(session_id)
        self._active_sessions.pop(session_id, None)

    async def status_all(self) -> dict:
        """Get status of all active sessions."""
        statuses = {}
        for sid, config in self._active_sessions.items():
            info = await self.orchestrator.get_session_status(sid)
            statuses[sid] = {
                "name": config.name,
                "status": info.status.value,
                "worker_id": info.worker_id,
                "pid": info.pid,
            }
        return statuses


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

async def main(config_path: str) -> None:
    """Main entrypoint: read config, create orchestrator, launch sessions."""
    # Load config
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_file) as f:
        raw_config = yaml.safe_load(f)

    # Parse session configs
    shared_api_key = os.getenv(
        "ANTHROPIC_API_KEY",
        raw_config.get("defaults", {}).get("anthropic_api_key", ""),
    )
    configs = build_multi_session_configs(raw_config, shared_api_key=shared_api_key)

    if not configs:
        logger.error("No valid session configs found")
        sys.exit(1)

    logger.info(f"Parsed {len(configs)} session configs from {config_path}")

    # Create orchestrator
    orchestrator = create_orchestrator()

    # Launch
    launcher = SessionLauncher(orchestrator)

    try:
        launched = await launcher.launch_all(configs)
        logger.info(f"All sessions launched. Running {len(launched)} bots.")

        # Keep running until interrupted
        logger.info("Press Ctrl+C to terminate all sessions")
        while True:
            await asyncio.sleep(30)
            statuses = await launcher.status_all()
            running = sum(1 for s in statuses.values() if s["status"] == "running")
            logger.info(f"Status: {running}/{len(statuses)} sessions running")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await launcher.terminate_all()
        await orchestrator.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Launch Claude Code + Discord sessions")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to sessions YAML config file",
    )
    args = parser.parse_args()

    asyncio.run(main(args.config))
