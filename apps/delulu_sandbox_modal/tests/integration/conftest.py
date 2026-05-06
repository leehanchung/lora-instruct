"""Shared fixtures for integration tests that hit the deployed Modal sandbox.

These tests call real Modal functions (``run_claude_code``,
``provision_workspace``, ``commit_workspace``) and exercise real
infrastructure — volumes, secrets, images, Claude Code CLI. They
cost real compute and Claude tokens, so they're gated behind the
``integration`` pytest marker and excluded from the default
``pytest`` run.

Prerequisites:
  - ``modal`` CLI authenticated (``modal setup`` or MODAL_TOKEN_* env vars)
  - The ``discord-orchestrator`` app deployed (``modal deploy ...``)
  - The ``claude-oauth`` and ``github-pat`` Modal secrets created

Run:
  cd apps/delulu_sandbox_modal
  uv run pytest tests/integration/ -v -m integration --timeout=180

The ``-m integration`` flag is required because the package's
``addopts = "-m 'not integration'"`` deselects this directory by
default — without it the suite collects zero tests and exits 0.
"""

from __future__ import annotations

import time

import modal
import pytest


# ── Markers ──────────────────────────────────────────────────
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests that call deployed Modal functions "
        "(deselect with '-m \"not integration\"')",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-apply the ``integration`` marker to every test in this directory."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ── Constants ────────────────────────────────────────────────
APP_NAME = "discord-orchestrator"
TEST_REPO_URL = "https://github.com/leehanchung/SMILE-factory.git"
TEST_REF = "main"


# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture(scope="module")
def run_claude_code_fn() -> modal.Function:
    """Handle to the deployed ``run_claude_code`` Modal function."""
    return modal.Function.from_name(APP_NAME, "run_claude_code")


@pytest.fixture(scope="module")
def provision_workspace_fn() -> modal.Function:
    """Handle to the deployed ``provision_workspace`` Modal function."""
    return modal.Function.from_name(APP_NAME, "provision_workspace")


@pytest.fixture(scope="module")
def commit_workspace_fn() -> modal.Function:
    """Handle to the deployed ``commit_workspace`` Modal function."""
    return modal.Function.from_name(APP_NAME, "commit_workspace")


@pytest.fixture()
def unique_thread_id() -> int:
    """Return a unique thread ID for test isolation.

    Uses the current time in nanoseconds — good enough for test
    isolation since no two test invocations will share the same ns
    timestamp. The workspace at ``/vol/workspaces/<thread_id>``
    is created fresh and not cleaned up (cleanup is a separate
    concern — the volume has plenty of space and stale test
    workspaces are harmless).
    """
    return int(time.time_ns())
