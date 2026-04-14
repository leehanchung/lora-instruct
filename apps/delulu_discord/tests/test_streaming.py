"""Unit tests for the live-status renderer in delulu_discord.streaming.

Two groups of tests here:

1. ``_render`` — a pure function from a list of event dicts to the
   markdown body of the status message.
2. ``LiveStatus`` edit plumbing — regression coverage for the
   kwargs we pass to ``Message.edit``. The initial Commit 3 ship
   crashed in production because ``suppress_embeds=True`` is valid
   on ``Messageable.send`` but not on ``Message.edit`` in
   discord.py 2.4, and the flush loop's ``TypeError`` propagated
   out of ``finalize_done`` and stranded every run on the
   "Thinking..." placeholder.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from delulu_discord.streaming import (
    DISCORD_MESSAGE_LIMIT,
    INITIAL_PLACEHOLDER,
    LiveStatus,
    _render,
    render_done,
)


def _tool_use(tool: str, summary: str) -> dict:
    return {"type": "tool_use", "tool": tool, "summary": summary}


def _tool_result(tool: str, *, ok: bool = True) -> dict:
    return {"type": "tool_result", "tool": tool, "ok": ok, "summary": ""}


def test_empty_transcript_shows_placeholder() -> None:
    assert _render([]) == INITIAL_PLACEHOLDER


def test_single_tool_use_without_result_shows_no_marker() -> None:
    rendered = _render([_tool_use("Read", "`src/app.py`")])
    assert "🔧 Read `src/app.py`" in rendered
    assert "✓" not in rendered
    assert "✗" not in rendered


def test_tool_use_with_ok_result_shows_check() -> None:
    rendered = _render([_tool_use("Read", "`src/app.py`"), _tool_result("Read", ok=True)])
    assert "🔧 Read `src/app.py` ✓" in rendered


def test_tool_use_with_error_result_shows_cross() -> None:
    rendered = _render([_tool_use("Bash", "`false`"), _tool_result("Bash", ok=False)])
    assert "🔧 Bash `false` ✗" in rendered


def test_text_event_appends_writing_marker() -> None:
    rendered = _render([_tool_use("Read", "`a.py`"), {"type": "text", "text": "hi"}])
    assert "✍️  Writing response..." in rendered
    # Writing marker is the last line
    assert rendered.splitlines()[-1].startswith("✍️")


def test_thinking_block_replaces_default_header_with_spoiler() -> None:
    rendered = _render(
        [
            {"type": "thinking", "text": "considering the approach"},
            _tool_use("Read", "`x.py`"),
        ]
    )
    lines = rendered.splitlines()
    assert lines[0].startswith("||🧠 Reasoning:")
    assert lines[0].endswith("||")
    assert "considering the approach" in lines[0]
    # Default placeholder should NOT appear when thinking is present
    assert INITIAL_PLACEHOLDER not in rendered


def test_multiple_thinking_blocks_keeps_latest_only() -> None:
    rendered = _render(
        [
            {"type": "thinking", "text": "first thought"},
            {"type": "thinking", "text": "second thought"},
        ]
    )
    assert "second thought" in rendered
    assert "first thought" not in rendered


def test_long_thinking_preview_is_truncated() -> None:
    long_thought = "a" * 500
    rendered = _render([{"type": "thinking", "text": long_thought}])
    assert "…||" in rendered
    # Spoiler wrapper + emoji + "Reasoning: " + up to 150 chars of thought + "…"
    assert len(rendered) < 250


def test_fifo_matching_of_same_tool_called_twice() -> None:
    """Two sequential Read calls should get their ticks matched in order."""
    rendered = _render(
        [
            _tool_use("Read", "`first.py`"),
            _tool_use("Read", "`second.py`"),
            _tool_result("Read", ok=True),
            _tool_result("Read", ok=False),
        ]
    )
    assert "🔧 Read `first.py` ✓" in rendered
    assert "🔧 Read `second.py` ✗" in rendered


def test_tool_result_without_matching_use_is_ignored() -> None:
    """Rogue tool_result with no prior tool_use shouldn't crash or add junk."""
    rendered = _render([_tool_result("Read", ok=True)])
    assert "✓" not in rendered
    assert "✗" not in rendered


def test_full_progression_matches_plan_example() -> None:
    transcript = [
        {"type": "thinking", "text": "OK, let me read those files"},
        _tool_use("Read", "`src/app.py`"),
        _tool_result("Read", ok=True),
        _tool_use("Read", "`src/handlers.py`"),
        _tool_result("Read", ok=True),
        _tool_use("Grep", "`handle_channel_message`"),
        {"type": "text", "text": "Based on the code..."},
    ]
    rendered = _render(transcript)
    lines = rendered.splitlines()
    assert lines[0].startswith("||🧠 Reasoning:")
    assert "🔧 Read `src/app.py` ✓" in rendered
    assert "🔧 Read `src/handlers.py` ✓" in rendered
    # Grep is still pending — no marker
    assert "🔧 Grep `handle_channel_message`" in rendered
    assert "🔧 Grep `handle_channel_message` ✓" not in rendered
    assert lines[-1].startswith("✍️")


def test_overflow_truncates_oldest_tool_lines() -> None:
    # 300 tool calls with long paths easily blow past 2000 chars
    transcript: list[dict] = [
        _tool_use("Read", f"`very/long/path/to/source/file_{i:04d}.py`") for i in range(300)
    ]
    rendered = _render(transcript)
    assert len(rendered) <= DISCORD_MESSAGE_LIMIT
    assert "earlier tool calls truncated" in rendered
    # Newest tool call must still be visible
    assert "file_0299.py" in rendered


def test_render_done_formatting() -> None:
    assert render_done(5, 8400) == "✅ Done • 5 tools • 8.4s"
    assert render_done(0, 0) == "✅ Done • 0 tools • 0.0s"
    assert render_done(1, 1234) == "✅ Done • 1 tools • 1.2s"


# ── LiveStatus edit kwargs (regression) ─────────────────────────
#
# Commit 3 shipped with ``status_msg.edit(..., suppress_embeds=True)``,
# which raises ``TypeError`` in discord.py 2.4 (``Message.edit`` does
# not accept it — only ``Messageable.send`` does). The resulting
# crash propagated out of ``finalize_done`` and left every run stuck
# on the initial "Thinking..." placeholder. These tests lock in the
# contract so the next accidental re-introduction fails in CI rather
# than in prod.


def _fake_message() -> MagicMock:
    msg = MagicMock()
    msg.edit = AsyncMock()
    return msg


async def test_flush_once_edit_kwargs_are_valid_for_message_edit() -> None:
    msg = _fake_message()
    live = LiveStatus(msg)
    live.push({"type": "tool_use", "tool": "Read", "summary": "`a.py`"})
    await live._flush_once()

    msg.edit.assert_called_once()
    call_kwargs = msg.edit.call_args.kwargs
    assert "suppress_embeds" not in call_kwargs, (
        "Message.edit does not accept suppress_embeds — set it on the "
        "initial thread.send instead; the flag sticks across edits."
    )
    assert "content" in call_kwargs
    assert "allowed_mentions" in call_kwargs


async def test_safe_edit_kwargs_are_valid_for_message_edit() -> None:
    """Used by finalize_done / finalize_error — same kwarg contract."""
    msg = _fake_message()
    live = LiveStatus(msg)
    await live._safe_edit("some content")

    msg.edit.assert_called_once()
    call_kwargs = msg.edit.call_args.kwargs
    assert "suppress_embeds" not in call_kwargs
    assert call_kwargs["content"] == "some content"


async def test_safe_edit_swallows_unexpected_errors() -> None:
    """If a future kwarg mismatch re-introduces a TypeError on edit,
    _safe_edit must NOT propagate it — otherwise finalize_done crashes
    the handler the way Commit 3's initial ship did."""
    msg = _fake_message()
    msg.edit.side_effect = TypeError("unexpected keyword argument 'foo'")
    live = LiveStatus(msg)

    # Must not raise — the whole point of _safe_edit is to log and move on.
    await live._safe_edit("content")
    msg.edit.assert_called_once()


async def test_flush_once_skips_edit_when_content_unchanged() -> None:
    msg = _fake_message()
    live = LiveStatus(msg)
    live.push({"type": "text", "text": "hello"})

    await live._flush_once()
    assert msg.edit.call_count == 1

    # No new events → _render output is unchanged → no redundant edit.
    await live._flush_once()
    assert msg.edit.call_count == 1
