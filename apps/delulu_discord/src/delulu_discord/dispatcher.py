"""Client-side dispatcher — calls the deployed Modal function from the bot.

This file was historically named ``sandbox.py`` and lived under
``modal_dispatch/``, but it has always been a *client-side* wrapper around
``modal.Function.from_name(...).remote()``. It never runs inside a sandbox.
The rename makes that boundary explicit: nothing in this module ships to
Modal; it only dispatches to a Modal App that was deployed from the sibling
``delulu_sandbox_modal`` package.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

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
        thread_id: int,
        prompt: str,
        *,
        repo_url: str | None = None,
        ref: str = "HEAD",
        resume: bool = False,
        attachments: list[tuple[str, bytes]] | None = None,
        message_id: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Dispatch a task to Modal and stream events as they arrive.

        This is an **async generator**: callers iterate with
        ``async for event in dispatcher.run_task(...)``. Each event is a
        plain dict matching one of the shapes declared in
        ``delulu_sandbox_modal.events`` (the bot cannot import the
        TypedDicts across the app boundary, so they're typed as
        ``dict[str, Any]`` here). The terminal event is ``done`` on a
        clean run or ``error`` on a nonzero Claude Code exit.

        Workspace derivation moved to the sandbox side in Phase 1 of
        the repo-provisioning rollout. The bot now passes
        ``thread_id`` (always) and optionally ``repo_url`` / ``ref``;
        the sandbox's ``provision_workspace`` Modal function (or its
        no-repo fast path) materializes the workspace and the run
        proceeds against it.

        Uses Modal's ``.remote_gen.aio`` so the stream is consumed
        directly on the event loop — no ``run_in_executor`` dance.
        """
        logger.info(
            "dispatch.start",
            session_id=session_id,
            thread_id=thread_id,
            repo_url=repo_url,
            ref=ref,
            resume=resume,
            attachment_count=len(attachments) if attachments else 0,
        )

        event_count = 0
        async for event in self._fn.remote_gen.aio(
            session_id=session_id,
            prompt=prompt,
            thread_id=thread_id,
            repo_url=repo_url,
            ref=ref,
            resume=resume,
            attachments=attachments or [],
            message_id=message_id,
        ):
            event_count += 1
            yield event

        logger.info(
            "dispatch.complete",
            session_id=session_id,
            event_count=event_count,
        )
