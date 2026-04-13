"""Sandbox dispatcher — called from the bot process to invoke the deployed
Modal function. The function itself lives in app.py; see the comment there
for why the decoration can't live here."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import modal
import structlog

if TYPE_CHECKING:
    from src.config import Settings

logger = structlog.get_logger()


class SandboxDispatcher:
    """Async wrapper around the deployed Modal function."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Modal 1.x requires an explicit Function.from_name lookup when
        # dispatching to a deployed function from a long-lived client.
        self._fn = modal.Function.from_name(
            settings.modal_app_name,
            "run_claude_code",
        )

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

        # .remote() is synchronous — run in executor to not block the bot
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._fn.remote(
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
