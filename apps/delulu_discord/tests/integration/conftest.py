"""Shared fixtures + marker setup for bot-side integration tests.

These tests cross the Modal RPC boundary (the dispatcher calls
deployed sandbox functions) and burn real compute, so they're
gated behind the ``integration`` pytest marker and excluded from
the default ``pytest`` run via the ``addopts`` in
``pyproject.toml``. CI overrides that with ``-m integration``.

Prerequisites:
  - ``modal`` CLI authenticated (``modal setup`` or MODAL_TOKEN_* env vars)
  - The ``discord-orchestrator`` app deployed (``modal deploy ...``)
  - The ``claude-oauth`` and ``github-pat`` Modal secrets created

Run:
  cd apps/delulu_discord
  uv run pytest tests/integration/ -v -m integration --timeout=180
"""

from __future__ import annotations

import pytest


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
