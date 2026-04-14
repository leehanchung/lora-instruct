"""Discord event handlers — message routing, thread creation, result posting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord
import structlog

if TYPE_CHECKING:
    from delulu_discord.dispatcher import SandboxDispatcher
    from delulu_discord.session_manager import SessionManager
    from delulu_discord.settings import Settings

logger = structlog.get_logger()


class MessageHandler:
    """Handles incoming Discord messages and dispatches Claude Code tasks."""

    def __init__(
        self,
        settings: Settings,
        session_manager: SessionManager,
        dispatcher: SandboxDispatcher,
    ) -> None:
        self.settings = settings
        self.sessions = session_manager
        self.dispatcher = dispatcher

    async def handle_channel_message(self, message: discord.Message, prompt: str) -> None:
        """New @-mention in a channel → create thread, dispatch task."""
        thread_name = prompt[:50].strip() or "Claude Code task"
        thread = await message.create_thread(name=thread_name)

        session = self.sessions.create_session(thread.id)
        attachments = await _download_attachments(message)

        logger.info(
            "task.new",
            thread_id=thread.id,
            session_id=session.session_id,
            prompt_preview=prompt[:80],
            attachment_count=len(attachments),
        )

        await self._dispatch_and_respond(
            thread, session, prompt, attachments, message.id, resume=False
        )

    async def handle_thread_reply(self, message: discord.Message, prompt: str) -> None:
        """Reply inside an existing thread → resume or start new session."""
        thread = message.channel
        assert isinstance(thread, discord.Thread)

        session, is_new = self.sessions.get_or_create(thread.id)
        resume = not is_new
        attachments = await _download_attachments(message)

        logger.info(
            "task.reply",
            thread_id=thread.id,
            session_id=session.session_id,
            resume=resume,
            attachment_count=len(attachments),
        )

        await self._dispatch_and_respond(
            thread, session, prompt, attachments, message.id, resume=resume
        )

    async def _dispatch_and_respond(
        self,
        thread: discord.Thread,
        session,
        prompt: str,
        attachments: list[tuple[str, bytes]],
        message_id: int,
        *,
        resume: bool,
    ) -> None:
        """Run Claude Code in a Modal sandbox and post the result.

        ``dispatcher.run_task`` is now an async generator that yields
        event dicts. This method drains the stream, picks the final
        text out of the terminal ``done`` / ``error`` event, and still
        posts a single message at the end — the live status message
        and 1-edit/sec flush loop land in a follow-up commit.
        """
        final_text = ""
        error_message: str | None = None
        async with thread.typing():
            try:
                async for event in self.dispatcher.run_task(
                    session_id=session.session_id,
                    workspace_path=session.workspace_path,
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
            except Exception:
                logger.exception("task.failed", session_id=session.session_id)
                await thread.send("Something went wrong running that task. Check the logs.")
                return

        if error_message:
            output = (
                f"{final_text}\n\n[error]\n{error_message}".strip()
                if final_text
                else f"[error] {error_message}"
            )
        else:
            output = final_text

        await self._post_result(thread, output)

    async def _post_result(self, thread: discord.Thread, output: str) -> None:
        """Post output to thread, falling back to file upload if too long."""
        if not output.strip():
            await thread.send("*(Claude Code produced no output)*")
            return

        if len(output) <= self.settings.max_output_length:
            # `suppress_embeds=True` stops Discord from auto-unfurling URLs,
            # and `allowed_mentions` prevents Claude's output from accidentally
            # pinging @everyone/@here or specific users/roles.
            await thread.send(
                output,
                allowed_mentions=discord.AllowedMentions.none(),
                suppress_embeds=True,
            )
        else:
            # Upload as a text file
            file = discord.File(
                fp=__import__("io").BytesIO(output.encode()),
                filename="claude-output.txt",
            )
            await thread.send("Output was too long for a message:", file=file)


async def _download_attachments(message: discord.Message) -> list[tuple[str, bytes]]:
    """Download every attachment on a Discord message as (filename, bytes).

    Claude Code inside the sandbox only sees the text prompt, so anything the
    user attached has to be fetched here and shipped across to Modal as bytes.
    """
    out: list[tuple[str, bytes]] = []
    for att in message.attachments:
        try:
            data = await att.read()
        except Exception:
            logger.exception(
                "attachment.download_failed",
                filename=att.filename,
                size=att.size,
            )
            continue
        out.append((att.filename, data))
    return out
