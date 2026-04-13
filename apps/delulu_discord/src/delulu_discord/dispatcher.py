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
        """Dispatch a task to Modal and return the output.

        The Modal function is now a generator that yields event dicts —
        this method collects the stream, finds the terminal
        ``done`` / ``error`` event, and returns the final text as a
        plain string so the rest of the bot keeps its current shape.
        The async-generator driver that actually streams events to
        Discord will replace this in a follow-up commit.
        """
        logger.info(
            "dispatch.start",
            session_id=session_id,
            resume=resume,
            attachment_count=len(attachments) if attachments else 0,
        )

        # .remote_gen() is a synchronous iterator — drain it in an executor
        # so the bot's event loop stays responsive.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._collect_events(
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

    def _collect_events(
        self,
        *,
        session_id: str,
        workspace_path: str,
        prompt: str,
        resume: bool,
        attachments: list[tuple[str, bytes]],
        message_id: int | None,
    ) -> str:
        """Drain the Modal generator and return the final text.

        On an ``error`` event this appends the error message to whatever
        partial text was produced, mirroring the old ``run_claude_code``
        behaviour of tacking ``[stderr]`` onto nonzero-exit output.
        """
        final_text = ""
        error_message: str | None = None
        for event in self._fn.remote_gen(
            session_id=session_id,
            workspace_path=workspace_path,
            prompt=prompt,
            resume=resume,
            attachments=attachments,
            message_id=message_id,
        ):
            etype = event.get("type") if isinstance(event, dict) else None
            if etype == "done":
                final_text = event.get("final_text") or final_text
            elif etype == "error":
                error_message = event.get("message") or "unknown error"
        if error_message:
            return (
                f"{final_text}\n\n[error]\n{error_message}".strip()
                if final_text
                else f"[error] {error_message}"
            )
        return final_text
