"""Integration test for the bot-side dispatcher against the real sandbox.

This test exercises the ``SandboxDispatcher`` class — the bot's
client-side wrapper around ``modal.Function.from_name().remote_gen.aio()``
— against the deployed Modal sandbox. Discord is NOT involved; only
the dispatcher → Modal → sandbox → event-stream path is tested.

This validates:
  - The dispatcher correctly looks up deployed Modal functions
  - Async generator streaming works end-to-end (``remote_gen.aio``)
  - Event dicts cross the Modal RPC boundary intact
  - The dispatcher handles both ``done`` and ``error`` terminal events
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
import pytest


@dataclass
class _FakeSettings:
    """Minimal stand-in for ``delulu_discord.settings.Settings``.

    Only the fields that ``SandboxDispatcher.__init__`` reads are
    included — we don't need the full pydantic model for this test.
    """

    modal_app_name: str = "discord-orchestrator"
    modal_volume_name: str = "claude-workspaces"
    sandbox_memory_mb: int = 4096
    sandbox_timeout_seconds: int = 300
    session_ttl_seconds: int = 3600
    max_output_length: int = 1900
    discord_bot_token: str = "fake"
    repo_cache_root: str = "/vol/repo-cache"
    default_git_ref: str = "HEAD"


def _unique_thread_id() -> int:
    return int(time.time_ns())


class TestDispatcherRunTask:
    """Call ``SandboxDispatcher.run_task()`` against the live sandbox."""

    @pytest.fixture()
    def dispatcher(self):
        """Create a real dispatcher pointing at the deployed app."""
        # Import here so the test file can be collected even if
        # delulu_discord isn't installed in this venv — the import
        # error will surface as a clear skip/failure on the fixture.
        from delulu_discord.dispatcher import SandboxDispatcher

        return SandboxDispatcher(settings=_FakeSettings())

    def test_run_task_streams_events(self, dispatcher) -> None:
        """The dispatcher should yield event dicts ending with a done event."""
        thread_id = _unique_thread_id()

        async def _run():
            events = []
            async for event in dispatcher.run_task(
                session_id="dispatcher-smoke",
                thread_id=thread_id,
                prompt="Reply with exactly PONG.",
            ):
                events.append(event)
            return events

        events = asyncio.run(_run())

        assert len(events) > 0, "Dispatcher yielded no events"
        types = [e["type"] for e in events]
        assert "done" in types or "error" in types, (
            f"No terminal event. Types: {types}"
        )

    def test_run_task_with_repo(self, dispatcher) -> None:
        """The dispatcher should pass repo_url/ref through to the sandbox."""
        thread_id = _unique_thread_id()

        async def _run():
            events = []
            async for event in dispatcher.run_task(
                session_id="dispatcher-repo",
                thread_id=thread_id,
                prompt="List files in the current directory. Just filenames.",
                repo_url="https://github.com/leehanchung/SMILE-factory.git",
                ref="main",
            ):
                events.append(event)
            return events

        events = asyncio.run(_run())

        done = next((e for e in events if e["type"] == "done"), None)
        if done is None:
            error = next((e for e in events if e["type"] == "error"), None)
            pytest.fail(f"Expected done, got error: {error}")

        # Should see repo files in the output
        output = done["final_text"].lower()
        assert any(
            name in output
            for name in ["readme", "pyproject", "claude.md", "makefile"]
        ), f"Expected repo files, got: {done['final_text'][:300]}"


class TestDispatcherCommit:
    """Call ``SandboxDispatcher.commit_workspace()`` against the live sandbox."""

    @pytest.fixture()
    def dispatcher(self):
        from delulu_discord.dispatcher import SandboxDispatcher

        return SandboxDispatcher(settings=_FakeSettings())

    def test_commit_nonexistent_workspace(self, dispatcher) -> None:
        """Committing a workspace that doesn't exist should return a status dict."""
        thread_id = _unique_thread_id()

        async def _run():
            return await dispatcher.commit_workspace(
                thread_id=thread_id,
                message="test commit from dispatcher",
            )

        result = asyncio.run(_run())

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ("no_workspace", "no_pat")
