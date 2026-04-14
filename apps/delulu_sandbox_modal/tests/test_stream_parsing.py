"""Unit tests for the stream-json parsers in delulu_sandbox_modal.app.

These exercise ``_flatten_stream_event``, ``_summarize_tool_input``, and
``_summarize_tool_result`` against **synthetic** stream-json lines that
match the documented Claude Code / Claude API content-block schema. They
do not prove Claude Code actually emits these exact shapes on the
current droplet version — that still needs a live capture — but they
do prove that *if* CC emits a content block of shape X, the parser
turns it into the bot-side event the renderer expects.

Most importantly: the thinking-block test is why this file exists.
The live bot doesn't currently show the ``||🧠 Reasoning: …||`` spoiler
because standard CC runs don't emit thinking blocks at all. This test
closes the loop: if CC ever does emit one, the parser surfaces it
correctly and the existing ``_render`` tests (in delulu_discord) prove
the bot will display it.
"""

from __future__ import annotations

from delulu_sandbox_modal.app import (
    _flatten_stream_event,
    _summarize_tool_input,
    _summarize_tool_result,
)

# ── _flatten_stream_event ────────────────────────────────────


def test_assistant_text_block_yields_text_event() -> None:
    line = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello there!"}],
        },
    }
    events = _flatten_stream_event(line, {})
    assert events == [{"type": "text", "text": "Hello there!"}]


def test_assistant_tool_use_block_populates_name_map() -> None:
    line = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "Read",
                    "input": {"file_path": "src/app.py"},
                }
            ],
        },
    }
    names: dict[str, str] = {}
    events = _flatten_stream_event(line, names)

    assert events == [{"type": "tool_use", "tool": "Read", "summary": "`src/app.py`"}]
    assert names == {"toolu_abc": "Read"}


def test_user_tool_result_maps_back_to_tool_name() -> None:
    names = {"toolu_abc": "Read"}
    line = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc",
                    "content": "file contents on one line",
                }
            ],
        },
    }
    events = _flatten_stream_event(line, names)
    assert events == [
        {
            "type": "tool_result",
            "tool": "Read",
            "ok": True,
            "summary": "file contents on one line",
        }
    ]


def test_tool_result_for_unknown_id_leaves_tool_empty() -> None:
    line = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_unknown",
                    "content": "mystery result",
                }
            ],
        },
    }
    events = _flatten_stream_event(line, {})
    assert len(events) == 1
    assert events[0]["type"] == "tool_result"
    assert events[0]["tool"] == ""
    assert events[0]["ok"] is True


def test_tool_result_is_error_marks_ok_false() -> None:
    names = {"toolu_x": "Bash"}
    line = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_x",
                    "is_error": True,
                    "content": "command not found",
                }
            ],
        },
    }
    events = _flatten_stream_event(line, names)
    assert events[0]["ok"] is False
    assert events[0]["tool"] == "Bash"


def test_assistant_thinking_block_yields_thinking_event() -> None:
    """The one that motivated this whole file.

    Standard CC doesn't emit ``{"type": "thinking"}`` content blocks —
    extended thinking is opt-in on the model side. But *if* CC ever
    surfaces one, ``_flatten_stream_event`` must turn it into a
    ``thinking`` event so the bot-side renderer can show the spoiler.
    """
    line = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me work through the approach step by step.",
                }
            ],
        },
    }
    events = _flatten_stream_event(line, {})
    assert events == [
        {
            "type": "thinking",
            "text": "Let me work through the approach step by step.",
        }
    ]


def test_assistant_message_with_mixed_blocks_yields_events_in_order() -> None:
    line = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "I'll read the file first."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "a.py"},
                },
            ],
        },
    }
    names: dict[str, str] = {}
    events = _flatten_stream_event(line, names)
    assert len(events) == 2
    assert events[0]["type"] == "text"
    assert events[1]["type"] == "tool_use"
    assert names == {"toolu_1": "Read"}


def test_system_and_unknown_types_produce_no_events() -> None:
    assert _flatten_stream_event({"type": "system", "subtype": "init"}, {}) == []
    assert _flatten_stream_event({"type": "wat", "foo": "bar"}, {}) == []
    assert _flatten_stream_event({}, {}) == []


def test_empty_text_block_is_skipped() -> None:
    line = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": ""}]},
    }
    assert _flatten_stream_event(line, {}) == []


def test_malformed_content_blocks_do_not_crash() -> None:
    """Non-dict content blocks should be skipped silently, not raise."""
    line = {
        "type": "assistant",
        "message": {
            "content": [
                "not-a-dict",
                None,
                {"type": "text", "text": "this one is fine"},
            ],
        },
    }
    events = _flatten_stream_event(line, {})
    assert events == [{"type": "text", "text": "this one is fine"}]


# ── _summarize_tool_input ────────────────────────────────────


def test_summarize_read_edit_write_uses_file_path() -> None:
    assert _summarize_tool_input("Read", {"file_path": "src/app.py"}) == "`src/app.py`"
    assert _summarize_tool_input("Edit", {"file_path": "a.md"}) == "`a.md`"
    assert _summarize_tool_input("Write", {"file_path": "b.py"}) == "`b.py`"


def test_summarize_bash_truncates_long_commands() -> None:
    long_cmd = "echo " + "x" * 200
    summary = _summarize_tool_input("Bash", {"command": long_cmd})
    assert summary.startswith("`")
    assert summary.endswith("`")
    assert len(summary) <= 82  # 80 chars + two backticks


def test_summarize_grep_and_glob_use_pattern() -> None:
    assert _summarize_tool_input("Grep", {"pattern": "foo.*bar"}) == "`foo.*bar`"
    assert _summarize_tool_input("Glob", {"pattern": "**/*.py"}) == "`**/*.py`"


def test_summarize_unknown_tool_returns_empty() -> None:
    assert _summarize_tool_input("WebFetch", {"url": "https://x.com"}) == ""


def test_summarize_missing_input_field_returns_empty() -> None:
    assert _summarize_tool_input("Read", {}) == ""
    assert _summarize_tool_input("Bash", {}) == ""


# ── _summarize_tool_result ────────────────────────────────────


def test_summarize_string_result_shows_first_line() -> None:
    result = "first line\nsecond line\nthird line"
    assert _summarize_tool_result(result) == "first line"


def test_summarize_long_string_truncates_with_ellipsis() -> None:
    result = "a" * 200
    summary = _summarize_tool_result(result)
    assert summary.endswith("…")
    # 80 chars + ellipsis
    assert len(summary) == 81


def test_summarize_list_of_content_blocks_returns_first_text() -> None:
    result = [{"type": "text", "text": "block text here"}]
    assert _summarize_tool_result(result) == "block text here"


def test_summarize_list_without_text_block_returns_empty() -> None:
    result = [{"type": "image", "source": {}}]
    assert _summarize_tool_result(result) == ""


def test_summarize_none_result_returns_empty() -> None:
    assert _summarize_tool_result(None) == ""


def test_summarize_empty_string_returns_empty() -> None:
    assert _summarize_tool_result("") == ""
    assert _summarize_tool_result("   \n  ") == ""
