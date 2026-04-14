"""Live-status message renderer and flush loop for the Discord bot.

Drives the single status message that gets edited in place as Claude
Code progresses through a run — coalescing events into at most one
Discord edit per second so we stay under the message-edit rate limit
of roughly 5 edits / 5 seconds / channel.

Everything here is pure ``asyncio`` / ``discord.py`` — nothing imports
from the ``delulu_sandbox_modal`` package, so the bot stays
independently deployable.
"""

from __future__ import annotations

import asyncio
from typing import Any

import discord
import structlog

logger = structlog.get_logger()

DISCORD_MESSAGE_LIMIT = 2000
FLUSH_INTERVAL_SECONDS = 1.0
THINKING_PREVIEW_LIMIT = 150
INITIAL_PLACEHOLDER = "💭 Thinking about your request..."


def render_done(num_tools: int, duration_ms: int) -> str:
    """Render the collapsed terminal 'done' status line."""
    duration_s = (duration_ms or 0) / 1000.0
    return f"✅ Done • {num_tools} tools • {duration_s:.1f}s"


def _render(
    transcript: list[dict[str, Any]],
    *,
    done_footer: str | None = None,
) -> str:
    """Render the running transcript into Discord markdown.

    Pure function — given the same list of events always produces the
    same output. Rules:

    - Empty transcript and no ``done_footer`` → initial
      "thinking..." placeholder.
    - Latest ``thinking`` block collapses into one spoiler line at the
      top (``||🧠 Reasoning: …||``).
    - Each ``tool_use`` becomes ``🔧 <Tool> <summary>`` with a trailing
      ``✓`` / ``✗`` if a matching ``tool_result`` followed it.
    - If any assistant ``text`` event is present AND the run isn't
      done yet, append an ``✍️ Writing response...`` marker at the
      bottom.
    - If ``done_footer`` is set, the run is finished: the
      ``✍️ Writing response...`` marker is dropped (it's no longer
      true) and ``done_footer`` is appended as the last line instead.
      This is how ``finalize_done`` keeps the live transcript visible
      as a permanent record of what Claude Code did, with a small
      ``✅ Done • N tools • Ts`` footer, rather than collapsing the
      whole message.
    - If the transcript overflows Discord's 2000-char limit, drop the
      oldest tool-call lines and prefix with a truncation marker.
    """
    if not transcript and done_footer is None:
        return INITIAL_PLACEHOLDER

    header = _render_header(transcript)
    tool_lines = _render_tool_lines(transcript)
    # The "writing response" marker only makes sense *during* the run;
    # once the run is done, the writing is also done, so suppress it.
    writing_marker = _render_writing_marker(transcript) if done_footer is None else None

    rendered = _assemble(header, tool_lines, writing_marker, done_footer)
    if len(rendered) <= DISCORD_MESSAGE_LIMIT:
        return rendered

    return _truncate_to_limit(header, tool_lines, writing_marker, done_footer)


def _render_header(transcript: list[dict[str, Any]]) -> str:
    thinking_texts = [
        e.get("text", "")
        for e in transcript
        if isinstance(e, dict) and e.get("type") == "thinking" and e.get("text")
    ]
    if not thinking_texts:
        return INITIAL_PLACEHOLDER

    latest = thinking_texts[-1].strip().replace("\n", " ")
    preview = (
        latest[:THINKING_PREVIEW_LIMIT] + "…" if len(latest) > THINKING_PREVIEW_LIMIT else latest
    )
    return f"||🧠 Reasoning: {preview}||"


def _render_tool_lines(transcript: list[dict[str, Any]]) -> list[str]:
    """Build the tool-call lines, matching tool_result → tool_use FIFO by name.

    We don't have a tool_use_id on the bot side (the sandbox's
    ``_flatten_stream_event`` only forwards the tool name), so two
    concurrent invocations of the same tool could end up matched out of
    order. This is cosmetic only — the final-text message is always
    assembled from ``DoneEvent.final_text``, so the user sees the
    correct answer regardless of how the status message paired ticks
    and crosses.
    """
    lines: list[str] = []
    pending_by_tool: dict[str, list[int]] = {}

    for event in transcript:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "tool_use":
            tool = event.get("tool", "?")
            summary = event.get("summary", "")
            line = f"🔧 {tool} {summary}".rstrip()
            pending_by_tool.setdefault(tool, []).append(len(lines))
            lines.append(line)
        elif etype == "tool_result":
            tool = event.get("tool", "")
            queue = pending_by_tool.get(tool)
            if queue:
                idx = queue.pop(0)
                marker = "✓" if event.get("ok", True) else "✗"
                lines[idx] = f"{lines[idx]} {marker}"

    return lines


def _render_writing_marker(transcript: list[dict[str, Any]]) -> str | None:
    has_text = any(isinstance(e, dict) and e.get("type") == "text" for e in transcript)
    return "✍️  Writing response..." if has_text else None


def _assemble(
    header: str,
    tool_lines: list[str],
    writing_marker: str | None,
    done_footer: str | None = None,
) -> str:
    parts = [header]
    parts.extend(tool_lines)
    if writing_marker:
        parts.append(writing_marker)
    if done_footer:
        parts.append(done_footer)
    return "\n".join(parts)


def _truncate_to_limit(
    header: str,
    tool_lines: list[str],
    writing_marker: str | None,
    done_footer: str | None = None,
) -> str:
    """Drop oldest tool lines until the assembled output fits.

    Always keeps at least the most recent tool line so the user sees
    what's currently running. Prefixes the remaining list with a
    ``… earlier tool calls truncated`` marker once anything is dropped.
    """
    kept = list(tool_lines)
    truncated = False
    while kept:
        candidate_tool_lines = ["🔧 … earlier tool calls truncated", *kept] if truncated else kept
        rendered = _assemble(header, candidate_tool_lines, writing_marker, done_footer)
        if len(rendered) <= DISCORD_MESSAGE_LIMIT:
            return rendered
        kept.pop(0)
        truncated = True

    # Pathological fallback: even an empty tool list overflows (huge
    # thinking preview). Hard-truncate the result so we never post
    # something Discord would reject outright.
    rendered = _assemble(header, [], writing_marker, done_footer)
    return rendered[:DISCORD_MESSAGE_LIMIT]


class LiveStatus:
    """Manages a single status message edited in place during a run.

    Usage::

        status_msg = await thread.send(INITIAL_PLACEHOLDER)
        live = LiveStatus(status_msg)
        live.start()
        try:
            async for event in dispatcher.run_task(...):
                live.push(event)
            await live.finalize_done(num_tools=..., duration_ms=...)
        except Exception:
            await live.finalize_error()
            raise
    """

    def __init__(
        self,
        status_msg: discord.Message,
        *,
        flush_interval: float = FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self.status_msg = status_msg
        self.flush_interval = flush_interval
        self.transcript: list[dict[str, Any]] = []
        self._last_rendered: str | None = None
        self._dirty = False
        self._stopped = False
        self._flush_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the background flush loop. Call once before ``push``."""
        if self._flush_task is not None:
            raise RuntimeError("LiveStatus.start() called twice")
        self._flush_task = asyncio.create_task(self._flush_loop())

    def push(self, event: dict[str, Any]) -> None:
        """Append an event and mark the transcript dirty for the next flush."""
        self.transcript.append(event)
        self._dirty = True

    async def finalize_done(self, *, num_tools: int, duration_ms: int) -> None:
        """Stop the flush loop and finalize the status message.

        The live transcript stays visible as a permanent record of
        what Claude Code did — tool calls, thinking preview, etc. —
        with a small ``✅ Done • N tools • Ts`` footer appended. The
        ``✍️ Writing response...`` marker is dropped at this point
        since it's no longer true.
        """
        await self._stop_flush()
        footer = render_done(num_tools, duration_ms)
        await self._safe_edit(_render(self.transcript, done_footer=footer))

    async def finalize_error(self) -> None:
        """Stop the flush loop and leave the status frozen on its last state.

        Does one final best-effort edit so the transcript the user sees
        in the status message reflects every event we managed to receive
        before the error, not the last rate-limited flush.
        """
        await self._stop_flush()
        await self._safe_edit(_render(self.transcript))

    async def _flush_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(self.flush_interval)
            except asyncio.CancelledError:
                return
            if not self._dirty or self._stopped:
                continue
            try:
                await self._flush_once()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("streaming.flush_failed")

    async def _flush_once(self) -> None:
        rendered = _render(self.transcript)
        if rendered == self._last_rendered:
            self._dirty = False
            return
        try:
            # NOTE: do NOT pass ``suppress_embeds`` here — ``Message.edit``
            # in discord.py 2.4 does not accept it and raises ``TypeError``
            # on every edit, which used to crash the whole flush loop
            # and then ``finalize_done``, leaving the status message
            # stuck on the initial placeholder forever. The embed-
            # suppressed flag is set once on the initial ``thread.send``
            # in ``handlers._dispatch_and_respond`` and inherited by
            # subsequent edits.
            await self.status_msg.edit(
                content=rendered,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            self._last_rendered = rendered
            self._dirty = False
        except discord.HTTPException as exc:
            if getattr(exc, "status", None) == 429:
                retry_after = float(getattr(exc, "retry_after", 5.0))
                logger.warning("streaming.rate_limited", retry_after=retry_after)
                await asyncio.sleep(retry_after)
            else:
                raise

    async def _stop_flush(self) -> None:
        self._stopped = True
        task = self._flush_task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        self._flush_task = None

    async def _safe_edit(self, content: str) -> None:
        # Same caveat as _flush_once: no ``suppress_embeds`` on edit.
        try:
            await self.status_msg.edit(
                content=content,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            self._last_rendered = content
        except discord.HTTPException:
            logger.exception("streaming.edit_failed")
        except Exception:
            # Anything else (TypeError from a future kwarg mismatch,
            # etc.) is a bug worth surfacing in logs but must not
            # propagate out of finalize_done and break the handler.
            logger.exception("streaming.edit_failed")
