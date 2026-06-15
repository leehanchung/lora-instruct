"""Per-server allowlist of repos the bot is permitted to provision.

Companion to ``repo_config.py``. Where ``RepoConfig`` stores
"this channel is bound to this repo", ``RepoAllowlist`` stores
"this Discord server is allowed to bind these repos at all" — the
access-control layer that prevents random users from pointing the
bot at huge or unrelated repositories. See the "Access control and
threat model" section of
``apps/delulu_discord/prd/repo-provisioning.md`` for the full
threat model.

Keyed by Discord ``guild_id``. Each entry tracks visibility
(``public`` / ``private``) at add time so the dispatch path can
decide whether the ``github-pat`` Modal secret is required without
re-hitting the network. See ``prd/private-repos.md`` for the
visibility marker design.

**Storage shape.** Each guild's value is now a dict keyed by
``owner/repo`` with a per-entry payload:

    {"alice/api-service": {"visibility": "public"},
     "alice/internal-api": {"visibility": "private"}}

v1 entries (pre-marker) were stored as a flat ``list[str]``. ``get``
detects the legacy shape on read and treats every entry as
``public`` — which is correct, since v1 explicitly rejected
private repos at add time.

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

VISIBILITY_PUBLIC = "public"
VISIBILITY_PRIVATE = "private"


class RepoAllowlist:
    """Modal-Dict-backed per-guild repo allowlist.

    All methods are **async** for the same reason as ``RepoConfig``:
    blocking ``modal.Dict`` calls stall discord.py's event loop.
    See the RepoConfig docstring for the full rationale.
    """

    def __init__(self) -> None:
        self._dict = modal.Dict.from_name(DICT_NAME, create_if_missing=True)

    async def get(self, guild_id: int) -> list[str]:
        """Return the list of allowed ``owner/repo`` short forms.

        Order is the dict's insertion order. Visibility is dropped
        on this projection — call ``get_visibility`` (or
        ``list_with_visibility``) when you need it. The bare-list
        view exists for backwards compatibility with autocomplete +
        ``contains`` callers that don't care about visibility.
        """
        raw = await self._dict.get.aio(guild_id)
        return [owner_repo for owner_repo, _ in _iter_entries(raw)]

    async def list_with_visibility(self, guild_id: int) -> list[tuple[str, str]]:
        """Return ``(owner_repo, visibility)`` tuples for the guild.

        Used by ``/admin_listrepos`` to surface the public/private
        split. Public if the entry is missing a marker (legacy v1
        rows) or explicitly tagged ``public``.
        """
        raw = await self._dict.get.aio(guild_id)
        return list(_iter_entries(raw))

    async def get_visibility(self, guild_id: int, owner_repo: str) -> str | None:
        """Return ``"public"`` / ``"private"`` for an entry, or None if absent.

        Called from the dispatch path to decide whether to refuse-
        and-instruct on a missing PAT. Returning ``None`` means the
        entry isn't on the allowlist — caller should treat as a
        validation error.
        """
        raw = await self._dict.get.aio(guild_id)
        for entry, visibility in _iter_entries(raw):
            if entry == owner_repo:
                return visibility
        return None

    async def add(
        self,
        guild_id: int,
        owner_repo: str,
        *,
        visibility: str = VISIBILITY_PUBLIC,
    ) -> None:
        """Add (or refresh) an entry to a guild's allowlist.

        Idempotent on the ``owner_repo`` key — re-adding an existing
        entry overwrites the visibility marker, which is the right
        behavior when an admin re-runs ``/admin_addrepo`` after a
        repo flips public→private upstream.
        """
        if visibility not in (VISIBILITY_PUBLIC, VISIBILITY_PRIVATE):
            raise ValueError(f"unknown visibility: {visibility!r}")

        raw = await self._dict.get.aio(guild_id)
        current = {entry: vis for entry, vis in _iter_entries(raw)}
        current[owner_repo] = visibility

        await self._dict.put.aio(guild_id, current)
        logger.info(
            "repo_allowlist.add",
            guild_id=guild_id,
            owner_repo=owner_repo,
            visibility=visibility,
        )

    async def remove(self, guild_id: int, owner_repo: str) -> None:
        """Remove an entry from a guild's allowlist. No-op if not present.

        Note: does NOT retroactively unbind channels that were
        previously bound to the removed repo. Existing bindings in
        ``RepoConfig`` survive until explicitly ``/unsetrepo``'d. The
        next ``/setrepo`` in those channels will fail the allowlist
        check, so the recovery path is "rebind to an allowed repo or
        unbind manually."
        """
        raw = await self._dict.get.aio(guild_id)
        current = {entry: vis for entry, vis in _iter_entries(raw)}
        if owner_repo not in current:
            return
        del current[owner_repo]
        await self._dict.put.aio(guild_id, current)
        logger.info(
            "repo_allowlist.remove",
            guild_id=guild_id,
            owner_repo=owner_repo,
        )

    async def contains(self, guild_id: int, owner_repo: str) -> bool:
        """True iff ``owner_repo`` is on ``guild_id``'s allowlist."""
        return owner_repo in await self.get(guild_id)


def _iter_entries(raw: object) -> list[tuple[str, str]]:
    """Normalize the stored value to ``[(owner_repo, visibility)]`` tuples.

    Three accepted on-disk shapes:

    1. ``None`` → no entries.
    2. ``list[str]`` → legacy v1 rows; every entry is ``public``.
    3. ``dict[str, dict]`` → current shape; ``visibility`` field
       extracted from the per-entry payload (defaults to ``public``
       if missing for forward-compat).

    Anything else returns an empty list — fail closed rather than
    crash the dispatch path on a corrupted dict value.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [(entry, VISIBILITY_PUBLIC) for entry in raw if isinstance(entry, str)]
    if isinstance(raw, dict):
        out: list[tuple[str, str]] = []
        for entry, payload in raw.items():
            if not isinstance(entry, str):
                continue
            if isinstance(payload, dict):
                visibility = payload.get("visibility", VISIBILITY_PUBLIC)
            else:
                visibility = VISIBILITY_PUBLIC
            out.append((entry, visibility))
        return out
    return []
