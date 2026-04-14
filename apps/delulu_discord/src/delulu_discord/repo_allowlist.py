"""Per-server allowlist of repos the bot is permitted to provision.

Companion to ``repo_config.py``. Where ``RepoConfig`` stores
"this channel is bound to this repo", ``RepoAllowlist`` stores
"this Discord server is allowed to bind these repos at all" — the
access-control layer that prevents random users from pointing the
bot at huge or unrelated repositories. See the "Access control and
threat model" section of ``prd/repo-provisioning.md`` for the full
threat model.

Keyed by Discord ``guild_id``; values are lists of ``owner/repo``
short forms (e.g. ``["alice/api-service", "alice-org/shared-lib"]``).

Phase 2 ships the data store; Phase 4 wires it into the
``/admin_addrepo`` / ``/admin_removerepo`` / ``/admin_listrepos``
slash commands gated on Discord's ``MANAGE_GUILD`` permission.
Until then this class exists but has no callers in the bot's
runtime path.

**Concurrency note.** ``add()`` / ``remove()`` are read-modify-write
on the underlying ``modal.Dict`` value, which has a TOCTOU window
if two admins concurrently mutate the same guild's allowlist.
For v1 admin commands are rare (one human at a time), so this is
acceptable. If contention ever becomes real, the right fix is to
move the mutation into a Modal function with ``max_containers=1``
keyed on the guild ID — same pattern as ``provision_workspace``
in the sandbox app.
"""

from __future__ import annotations

import modal
import structlog

logger = structlog.get_logger()

DICT_NAME = "discord-orchestrator-allowlist"


class RepoAllowlist:
    """Modal-Dict-backed per-guild repo allowlist."""

    def __init__(self) -> None:
        self._dict = modal.Dict.from_name(DICT_NAME, create_if_missing=True)

    def get(self, guild_id: int) -> list[str]:
        """Return the allowlist for a guild (empty list if unset).

        Used by ``/setrepo`` autocomplete + validation, and by
        ``/admin_listrepos`` to show the current state.
        """
        return list(self._dict.get(guild_id) or [])

    def add(self, guild_id: int, owner_repo: str) -> None:
        """Add an entry to a guild's allowlist. Idempotent."""
        current = self.get(guild_id)
        if owner_repo in current:
            return
        current.append(owner_repo)
        self._dict[guild_id] = current
        logger.info(
            "repo_allowlist.add",
            guild_id=guild_id,
            owner_repo=owner_repo,
        )

    def remove(self, guild_id: int, owner_repo: str) -> None:
        """Remove an entry from a guild's allowlist. No-op if not present.

        Note: does NOT retroactively unbind channels that were
        previously bound to the removed repo. Existing bindings in
        ``RepoConfig`` survive until explicitly ``/unsetrepo``'d. The
        next ``/setrepo`` in those channels will fail the allowlist
        check, so the recovery path is "rebind to an allowed repo or
        unbind manually."
        """
        current = self.get(guild_id)
        if owner_repo not in current:
            return
        current.remove(owner_repo)
        self._dict[guild_id] = current
        logger.info(
            "repo_allowlist.remove",
            guild_id=guild_id,
            owner_repo=owner_repo,
        )

    def contains(self, guild_id: int, owner_repo: str) -> bool:
        """True iff ``owner_repo`` is on ``guild_id``'s allowlist."""
        return owner_repo in self.get(guild_id)
