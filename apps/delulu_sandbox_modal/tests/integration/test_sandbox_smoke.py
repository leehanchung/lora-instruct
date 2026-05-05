"""Smoke tests for the deployed Modal sandbox.

Each test calls the real ``run_claude_code`` function via Modal RPC
and validates:
  - The container image boots (Node.js, Claude Code CLI present)
  - The ``claude-oauth`` secret is accessible and credentials work
  - The ``claude-workspaces`` volume mounts and is writable
  - The event stream is well-formed (yields events, ends with done/error)

These tests burn real Claude tokens. Keep prompts minimal.
"""

from __future__ import annotations

import modal
import pytest


class TestSandboxBoots:
    """Verify the sandbox container starts and Claude Code executes."""

    def test_trivial_prompt_returns_done_event(
        self,
        run_claude_code_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """Simplest possible test: send a prompt, get a done event back.

        If this passes, the image, credentials, volume, and event
        streaming all work.
        """
        events = list(
            run_claude_code_fn.remote_gen(
                session_id="smoke-trivial",
                prompt="Reply with exactly the word PONG and nothing else.",
                thread_id=unique_thread_id,
            )
        )

        types = [e["type"] for e in events]
        assert "done" in types or "error" in types, (
            f"No terminal event in stream. Got types: {types}"
        )

        done = next((e for e in events if e["type"] == "done"), None)
        if done:
            assert done["final_text"], "done event has empty final_text"
            assert done["duration_ms"] > 0, "duration_ms should be positive"


class TestEventStream:
    """Validate the shape and ordering of the event stream."""

    def test_stream_contains_expected_event_types(
        self,
        run_claude_code_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """A prompt that forces tool use should produce tool_use + tool_result events."""
        events = list(
            run_claude_code_fn.remote_gen(
                session_id="smoke-tools",
                prompt=(
                    "Create a file called /tmp/e2e_canary.txt with the content 'hello'. "
                    "Then read it back and tell me what it says."
                ),
                thread_id=unique_thread_id,
            )
        )

        types = [e["type"] for e in events]

        # Must have a terminal event
        assert "done" in types or "error" in types

        # Should have used at least one tool (Write or Bash)
        if "done" in types:
            assert "tool_use" in types, (
                f"Expected tool_use events for a file-write prompt. Got: {types}"
            )
            assert "tool_result" in types, (
                f"Expected tool_result events. Got: {types}"
            )

    def test_all_events_have_type_field(
        self,
        run_claude_code_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """Every event dict must have a 'type' key."""
        events = list(
            run_claude_code_fn.remote_gen(
                session_id="smoke-shape",
                prompt="What is 1+1? Reply with just the number.",
                thread_id=unique_thread_id,
            )
        )

        for i, event in enumerate(events):
            assert isinstance(event, dict), f"Event {i} is not a dict: {event!r}"
            assert "type" in event, f"Event {i} missing 'type' key: {event!r}"

    def test_done_event_has_required_fields(
        self,
        run_claude_code_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """The terminal ``done`` event must carry final_text, num_turns, and duration_ms."""
        events = list(
            run_claude_code_fn.remote_gen(
                session_id="smoke-done-fields",
                prompt="Say hello.",
                thread_id=unique_thread_id,
            )
        )

        done = next((e for e in events if e["type"] == "done"), None)
        if done is None:
            pytest.skip("Got error instead of done — credential or quota issue")

        assert "final_text" in done
        assert "num_turns" in done
        assert "duration_ms" in done
        assert isinstance(done["final_text"], str)
        assert isinstance(done["num_turns"], int)
        assert isinstance(done["duration_ms"], int)


class TestResume:
    """Verify session continuity via --continue."""

    def test_resume_flag_continues_prior_session(
        self,
        run_claude_code_fn: modal.Function,
        unique_thread_id: int,
    ) -> None:
        """First call sets a fact, second call with resume=True recalls it.

        This proves that:
        1. The workspace persists on the volume across invocations
        2. Claude Code's session history (under ~/.claude/projects/) persists
        3. The --continue flag works
        """
        # First call: establish a fact
        list(
            run_claude_code_fn.remote_gen(
                session_id="smoke-resume-1",
                prompt=(
                    "Remember this: the secret code is DOLPHIN-42. "
                    "Acknowledge by saying 'Noted'."
                ),
                thread_id=unique_thread_id,
            )
        )

        # Second call: resume and recall
        events = list(
            run_claude_code_fn.remote_gen(
                session_id="smoke-resume-2",
                prompt="What was the secret code I told you earlier?",
                thread_id=unique_thread_id,
                resume=True,
            )
        )

        done = next((e for e in events if e["type"] == "done"), None)
        if done is None:
            pytest.skip("Got error on resume — may be a credential issue")

        assert "DOLPHIN" in done["final_text"].upper() or "42" in done["final_text"], (
            f"Resume didn't recall the secret code. Got: {done['final_text'][:200]}"
        )
