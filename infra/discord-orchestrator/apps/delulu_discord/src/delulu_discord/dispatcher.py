"""Client-side dispatcher — calls the deployed Modal function from the bot.

This file was historically named ``sandbox.py`` and lived under
``modal_dispatch/``, but it has always been a *client-side* wrapper around
``modal.Function.from_name(...).remote()``. It never runs inside a sandbox.
The rename makes that boundary explicit: nothing in this module ships to
Modal; it only dispatches to a Modal App that was deployed from the sibling
``delulu_sandbox_modal`` package.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import modal
import structlog

if TYPE_CHECKING:
    from delulu_discord.settings import Settings

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
        attachments: list[tuple[str, bytes]] | None = None,
        message_id: int | None = None,
    ) -> str:
        """Dispatch a task to Modal and return the output."""
        logger.info(
            "dispatch.start",
            session_id=session_id,
            resume=resume,
            attachment_count=len(attachments) if attachments else 0,
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
                attachments=attachments or [],
                message_id=message_id,
            ),
        )

        logger.info(
            "dispatch.complete",
            session_id=session_id,
            output_length=len(result),
        )
        return result
