"""Sandbox dispatcher — runs Claude Code tasks in Modal containers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import modal
import structlog

from src.modal_dispatch.app import anthropic_secret, app, sandbox_image, volume

if TYPE_CHECKING:
    from src.config import Settings

logger = structlog.get_logger()


# ── Modal Function (runs inside the sandbox) ─────────────────
@app.function(
    image=sandbox_image,
    volumes={"/vol": volume},
    secrets=[anthropic_secret],
    memory=4096,
    timeout=300,
)
def run_claude_code(
    session_id: str,
    workspace_path: str,
    prompt: str,
    *,
    resume: bool = False,
) -> str:
    """Execute a Claude Code task inside an ephemeral Modal sandbox.

    Args:
        session_id: Unique session identifier for Claude Code's --resume.
        workspace_path: Absolute path on the volume (e.g., /vol/workspaces/<id>).
        prompt: The user's prompt / instruction.
        resume: Whether to resume an existing Claude Code session.

    Returns:
        Claude Code's stdout output as a string.
    """
    import os
    import subprocess

    # Ensure workspace directory exists
    os.makedirs(workspace_path, exist_ok=True)

    # Build the claude command
    cmd = [
        "claude",
        "--print",             # non-interactive, print result
        "-p", prompt,          # the prompt
        "--output-format", "text",
    ]

    if resume:
        cmd.append("--resume")

    # Set Claude Code's session/project directory so --resume works across invocations
    env = {
        **os.environ,
        "CLAUDE_PROJECT_DIR": workspace_path,
    }

    logger.info(
        "sandbox.exec",
        session_id=session_id,
        resume=resume,
        workspace=workspace_path,
        prompt_preview=prompt[:80],
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=workspace_path,
        env=env,
        timeout=280,  # slightly under Modal's 300s timeout
    )

    if result.returncode != 0:
        logger.warning(
            "sandbox.nonzero_exit",
            session_id=session_id,
            returncode=result.returncode,
            stderr=result.stderr[:500],
        )

    # Commit volume changes so they persist for the next invocation
    volume.commit()

    output = result.stdout.strip()
    if result.returncode != 0 and result.stderr.strip():
        output += f"\n\n[stderr]\n{result.stderr.strip()}"

    return output


# ── Dispatcher (called from the bot process) ─────────────────
class SandboxDispatcher:
    """Async wrapper around the Modal function, called from the Discord bot."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run_task(
        self,
        session_id: str,
        workspace_path: str,
        prompt: str,
        *,
        resume: bool = False,
    ) -> str:
        """Dispatch a task to Modal and return the output."""
        logger.info(
            "dispatch.start",
            session_id=session_id,
            resume=resume,
        )

        # Modal's .remote() is synchronous — run in executor to not block the bot
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_claude_code.remote(
                session_id=session_id,
                workspace_path=workspace_path,
                prompt=prompt,
                resume=resume,
            ),
        )

        logger.info(
            "dispatch.complete",
            session_id=session_id,
            output_length=len(result),
        )
        return result
