"""Discord event handlers — message routing, thread creation, result posting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord
import structlog

if TYPE_CHECKING:
    from src.bot.session_manager import SessionManager
    from src.config import Settings
    from src.modal_dispatch.sandbox import SandboxDispatcher

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

    async def handle_channel_message(
        self, message: discord.Message, prompt: str
    ) -> None:
        """New @-mention in a channel → create thread, dispatch task."""
        thread_name = prompt[:50].strip() or "Claude Code task"
        thread = await message.create_thread(name=thread_name)

        session = self.sessions.create_session(thread.id)

        logger.info(
            "task.new",
            thread_id=thread.id,
            session_id=session.session_id,
            prompt_preview=prompt[:80],
        )

        await self._dispatch_and_respond(thread, session, prompt, resume=False)

    async def handle_thread_reply(
        self, message: discord.Message, prompt: str
    ) -> None:
        """Reply inside an existing thread → resume or start new session."""
        thread = message.channel
        assert isinstance(thread, discord.Thread)

        session, is_new = self.sessions.get_or_create(thread.id)
        resume = not is_new

        logger.info(
            "task.reply",
            thread_id=thread.id,
            session_id=session.session_id,
            resume=resume,
        )

        await self._dispatch_and_respond(thread, session, prompt, resume=resume)

    async def _dispatch_and_respond(
        self,
        thread: discord.Thread,
        session,
        prompt: str,
        *,
        resume: bool,
    ) -> None:
        """Run Claude Code in a Modal sandbox and post the result."""
        # Show typing indicator while working
        async with thread.typing():
            try:
                result = await self.dispatcher.run_task(
                    session_id=session.session_id,
                    workspace_path=session.workspace_path,
                    prompt=prompt,
                    resume=resume,
                )
            except Exception:
                logger.exception("task.failed", session_id=session.session_id)
                await thread.send("Something went wrong running that task. Check the logs.")
                return

        # Post result, respecting Discord's character limit
        await self._post_result(thread, result)

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
