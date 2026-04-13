"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config — reads from env vars or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Discord ──────────────────────────────────────────────
    discord_bot_token: str

    # ── Modal ────────────────────────────────────────────────
    modal_volume_name: str = "claude-workspaces"
    modal_app_name: str = "discord-orchestrator"
    sandbox_memory_mb: int = 4096
    sandbox_timeout_seconds: int = 300  # max time per Claude Code invocation

    # ── Session behavior ─────────────────────────────────────
    session_ttl_seconds: int = 3600  # 1 hour before session resets
    max_output_length: int = 1900  # Discord limit minus some margin
