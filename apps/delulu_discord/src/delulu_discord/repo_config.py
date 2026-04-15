"""Persistent channel→repo binding store, backed by a Modal Dict.

This is the data layer for the /setrepo and /unsetrepo slash
commands (wired up in Phase 3). The store maps Discord channel IDs
to a (repo_url, ref) tuple — when the bot dispatches a new thread
in a bound channel, it looks up the binding here and threads
``repo_url`` / ``ref`` through to the sandbox's
``provision_workspace``.

Lives on the bot side, not the sandbox side. The bot has Modal
client auth via ``/root/.modal.toml``, so it can talk to
``modal.Dict`` directly without going through a deployed function.

The Dict is created lazily on first use via ``create_if_missing=True``
— no out-of-band setup required when standing up a fresh deployment.
Phase 2 wires this class into MessageHandler but doesn't yet expose
any way to add bindings; that comes in Phase 3 with the /setrepo
slash command. Until then, ``get()`` always returns ``None`` and
the dispatch path falls through to the no-repo (general Q&A)
behavior — exactly the same shape as today's bot.
"""

from __future__ import annotations

import modal
import structlog

logger = structlog.get_logger()

DICT_NAME = "discord-orchestrator-repo-config"


class RepoConfig:
    """Modal-Dict-backed channel→(repo_url, ref) binding store.

    All methods are **async** because the bot runs on discord.py's
    asyncio event loop. Using ``modal.Dict``'s blocking dict-style
    API (``d.get(key)``, ``d[key] = value``, ``d.pop(key)``) from
    within a coroutine stalls the event loop for the duration of
    the Modal round-trip (~50–200ms per call), which freezes every
    other async task in the bot — gateway heartbeats, inbound
    messages, slash-command interactions. Modal surfaces this as
    ``AsyncUsageWarning``.

    The fix is to use modal.Dict's ``.aio`` variants, which return
    awaitables. Each blocking method on ``modal.Dict`` exposes an
    ``.aio`` attribute that's the async version of the same call —
    e.g. ``d.get.aio(key)`` returns a coroutine.
    """

    def __init__(self) -> None:
        self._dict = modal.Dict.from_name(DICT_NAME, create_if_missing=True)

    async def get(self, channel_id: int) -> tuple[str, str] | None:
        """Return ``(repo_url, ref)`` for a channel, or ``None`` if unbound.

        Used by the message handler at thread-creation time. Returning
        ``None`` is the "general Q&A mode" sentinel — the dispatch
        proceeds with an empty workspace and no git operations.
        """
        raw = await self._dict.get.aio(channel_id)
        if raw is None:
            return None
        # Stored as a small dict so the schema is self-describing in
        # Modal's UI and survives field additions.
        return raw["repo_url"], raw["ref"]

    async def set(self, channel_id: int, repo_url: str, ref: str = "HEAD") -> None:
        """Bind a channel to ``(repo_url, ref)``. Overwrites any existing binding."""
        await self._dict.put.aio(channel_id, {"repo_url": repo_url, "ref": ref})
        logger.info(
            "repo_config.set",
            channel_id=channel_id,
            repo_url=repo_url,
            ref=ref,
        )

    async def unset(self, channel_id: int) -> None:
        """Remove a channel's binding. No-op if not present."""
        # `pop` on modal.Dict raises KeyError if missing; swallow it
        # so callers don't have to special-case the no-binding path.
        try:
            await self._dict.pop.aio(channel_id)
            logger.info("repo_config.unset", channel_id=channel_id)
        except KeyError:
            pass
