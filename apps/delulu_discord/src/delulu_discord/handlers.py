"""Discord event handlers — message routing, thread creation, result posting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord
import structlog

from delulu_discord.repo_allowlist import VISIBILITY_PRIVATE
from delulu_discord.streaming import LiveStatus, _render

if TYPE_CHECKING:
    from delulu_discord.dispatcher import SandboxDispatcher
    from delulu_discord.repo_allowlist import RepoAllowlist
    from delulu_discord.repo_config import RepoConfig
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
        repo_config: RepoConfig,
        repo_allowlist: RepoAllowlist,
    ) -> None:
        self.settings = settings
        self.sessions = session_manager
        self.dispatcher = dispatcher
        self.repo_config = repo_config
        self.repo_allowlist = repo_allowlist

    async def handle_channel_message(self, message: discord.Message, prompt: str) -> None:
        """New @-mention in a channel → create thread, dispatch task.

        Looks up the channel's repo binding via ``RepoConfig.get()``
        — if the channel is bound, the new session is created with
        the (repo_url, ref) tuple, which the sandbox's
        ``provision_workspace`` will use to clone+worktree the repo.
        If unbound (the current default for every channel until
        Phase 3 ships ``/setrepo``), the session has ``repo_url=None``
        and the sandbox falls through to the empty-workspace
        general-Q&A path.
        """
        thread_name = prompt[:50].strip() or "Claude Code task"
        thread = await message.create_thread(name=thread_name)

        binding = await self.repo_config.get(message.channel.id)
        if binding is None:
            repo_url, ref = None, self.settings.default_git_ref
            is_private = False
        else:
            repo_url, ref = binding
            # Look up the repo's visibility on the server's allowlist.
            # Bound repos are *almost always* on the allowlist (because
            # /setrepo enforces it at bind time), but defensive: if a
            # repo was removed from the allowlist after binding, the
            # lookup returns None and we fall back to public — which
            # means the dispatch goes through unauthenticated and
            # fails cleanly on a private repo with the standard git
            # auth error rather than an opaque crash.
            owner_repo = _short_repo_from_url(repo_url)
            if message.guild is not None and owner_repo is not None:
                visibility = await self.repo_allowlist.get_visibility(message.guild.id, owner_repo)
                is_private = visibility == VISIBILITY_PRIVATE
            else:
                is_private = False

        session = self.sessions.create_session(
            thread.id,
            repo_url=repo_url,
            ref=ref,
            is_private=is_private,
        )
        attachments = await _download_attachments(message)

        logger.info(
            "task.new",
            thread_id=thread.id,
            session_id=session.session_id,
            prompt_preview=prompt[:80],
            attachment_count=len(attachments),
            repo_url=repo_url,
            ref=ref,
            is_private=is_private,
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
        """Run Claude Code and stream live progress into a status message.

        Posts an initial ``💭 Thinking…`` message, spawns a background
        flush loop that edits that message at most once per second as
        events arrive, collapses the status to ``✅ Done • N tools • Ts``
        when the stream ends, and then posts the final assistant text
        as a separate message (so it's findable in Discord search and
        isn't buried inside a long transcript).

        On error the status message freezes on its last rendered state
        and a separate ``⚠️`` message carries the error details.
        """
        # ``suppress_embeds`` is set here on the send — it's valid on
        # ``Messageable.send`` but NOT on ``Message.edit`` in discord.py
        # 2.4 (that's the crash that broke the initial Commit 3 ship).
        # The flag sticks across edits, so setting it once at post time
        # is enough to keep the live status from unfurling any URLs
        # that might appear in tool summaries.
        # Render the initial placeholder with the repo subtitle (if
        # bound) baked in, so the very first state the user sees is
        # already oriented to the repo. ``_render`` with an empty
        # transcript and no done_footer returns the placeholder; the
        # repo line is appended below it when ``repo_url`` is set.
        # The subsequent flush loop and finalize_done will continue
        # to pass repo_url/ref through _render so the subtitle stays
        # visible across the whole message lifecycle.
        initial_content = _render(
            [],
            repo_url=session.repo_url,
            ref=session.ref,
            is_private=session.is_private,
        )
        status_msg = await thread.send(
            initial_content,
            allowed_mentions=discord.AllowedMentions.none(),
            suppress_embeds=True,
        )
        live = LiveStatus(
            status_msg,
            repo_url=session.repo_url,
            ref=session.ref,
            is_private=session.is_private,
        )
        live.start()

        final_text = ""
        duration_ms = 0
        error_message: str | None = None

        try:
            async for event in self.dispatcher.run_task(
                session_id=session.session_id,
                thread_id=session.thread_id,
                prompt=prompt,
                repo_url=session.repo_url,
                ref=session.ref,
                is_private=session.is_private,
                resume=resume,
                attachments=attachments,
                message_id=message_id,
            ):
                live.push(event)
                etype = event.get("type") if isinstance(event, dict) else None
                if etype == "done":
                    final_text = event.get("final_text") or final_text
                    duration_ms = int(event.get("duration_ms") or 0)
                elif etype == "error":
                    error_message = event.get("message") or "unknown error"
        except Exception:
            logger.exception("task.failed", session_id=session.session_id)
            await live.finalize_error()
            await thread.send("Something went wrong running that task. Check the logs.")
            return

        num_tools = sum(
            1 for e in live.transcript if isinstance(e, dict) and e.get("type") == "tool_use"
        )

        if error_message:
            await live.finalize_error()
            await thread.send(
                f"⚠️ {error_message}",
                allowed_mentions=discord.AllowedMentions.none(),
                suppress_embeds=True,
            )
            return

        await live.finalize_done(num_tools=num_tools, duration_ms=duration_ms)
        await self._post_result(thread, final_text)

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


def _short_repo_from_url(repo_url: str) -> str | None:
    """Return ``owner/repo`` from a stored repo URL, or None if unparseable.

    Mirrors the logic in ``streaming._short_repo_name`` but returns
    ``None`` on failure (instead of falling back to the raw URL)
    because the caller uses the result as an allowlist key — a bad
    parse should skip the lookup, not query for a nonsense entry.
    """
    from urllib.parse import urlparse

    url = repo_url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4].rstrip("/")
    if url.startswith("git@"):
        _, _, path = url[len("git@") :].partition(":")
    else:
        path = urlparse(url).path.lstrip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return None


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
