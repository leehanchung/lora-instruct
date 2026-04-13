"""Modal app definition — image, volume, and secrets configuration.

This module defines the Modal infrastructure. The sandbox function in sandbox.py
uses these definitions to run Claude Code in ephemeral containers.
"""

from __future__ import annotations

import modal

# ── Modal App ────────────────────────────────────────────────
app = modal.App("discord-orchestrator")

# ── Persistent Volume ────────────────────────────────────────
# Shared across all sandboxes. Holds workspaces and Claude Code session history.
volume = modal.Volume.from_name("claude-workspaces", create_if_missing=True)

# ── Container Image ──────────────────────────────────────────
# Pre-bake Claude Code into the image so sandbox startup is fast.
sandbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .run_commands(
        # Install Node.js (required by Claude Code)
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        # Install Claude Code globally
        "npm install -g @anthropic-ai/claude-code",
    )
)

# ── Secrets ──────────────────────────────────────────────────
# Claude Code OAuth credentials (Pro/Max subscription). Seeded into the volume on
# first run, then refreshed-in-place on the volume so rotated refresh tokens
# survive across sandbox invocations.
#
# Create with:
#   modal secret create claude-oauth \
#     CLAUDE_CREDENTIALS_JSON="$(cat ~/.claude/.credentials.json)"
claude_oauth_secret = modal.Secret.from_name("claude-oauth")

# ── Register functions on the App ────────────────────────────
# This import has to live at the bottom: sandbox.py imports the names
# defined above, then applies @app.function to register run_claude_code.
# Without this, `modal deploy src/modal_dispatch/app.py` would register an
# empty App — the decorator never runs.
from src.modal_dispatch import sandbox  # noqa: E402, F401
