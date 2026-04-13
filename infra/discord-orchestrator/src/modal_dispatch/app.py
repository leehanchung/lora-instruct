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
# Anthropic API key — create with: modal secret create anthropic-key ANTHROPIC_API_KEY=sk-...
anthropic_secret = modal.Secret.from_name("anthropic-key")
