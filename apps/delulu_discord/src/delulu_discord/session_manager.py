"""SQLite-backed per-thread session store.

Replaces the previous in-memory ``dict[int, Session]`` so per-thread
state survives bot restarts. The deploy path on the VPS does
``docker rm`` + ``docker run`` (see the ``restart`` target in
``apps/delulu_discord/Makefile``); without persistence, every deploy
silently wiped the ``repo_url`` / ``ref`` / ``is_private`` that
``handle_channel_message`` had stitched onto each thread's session
from ``RepoConfig``. Subsequent ``handle_thread_reply`` calls (which
do *not* re-consult ``RepoConfig``) then fell through to the
empty-workspace path, surfacing as "the bot forgot my /setrepo."

Storage layout:

- A SQLite file at ``Settings.session_db_path`` (defaults to
  ``/data/sessions.db``, mounted from the ``disco-data`` named Docker
  volume in the Makefile's ``restart`` target). Named volumes survive
  ``docker rm``, which is what makes this fix work.
- One row per Discord thread, keyed on ``thread_id`` (PK).
- WAL journal mode for crash safety + concurrent reads.

Concurrency model:

- A single ``aiosqlite.Connection`` opened in ``connect()`` and
  closed in ``close()``. aiosqlite owns one background thread per
  connection and serializes every operation onto it; the bot is a
  single-process asyncio app, so one connection is plenty and avoids
  per-call thread-handoff cost on the hot path (every Discord
  message hits this).
- **One bot replica only.** WAL handles concurrent processes on the
  same host file, but two droplets writing to the same file would
  corrupt the DB — that's a Postgres-shaped problem if it ever comes
  up.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite
import structlog

logger = structlog.get_logger()

# Bumped when the schema changes in a way that requires a migration.
# Read on connect from ``schema_meta``; new columns can branch on it.
SCHEMA_VERSION = "1"
SCHEMA_VERSION_KEY = "schema_version"

# How recent ``last_active_at`` must be before ``get_session`` skips
# its persistence-side touch. The bot is on a single replica with an
# hour-long TTL, so a few seconds of drift on the persisted clock is
# invisible across restarts; meanwhile the in-memory ``Session`` is
# still updated, so TTL gating in this process stays exact. Trading
# this for an UPDATE+commit on every Discord message in an active
# thread is the load-bearing perf win in this module.
TOUCH_DEBOUNCE_SECONDS = 60

# Rows whose ``last_active_at`` is older than this on connect get
# swept by the startup janitor. The TTL window is an hour; anything
# 30 days idle is dead. Sweeping at connect (not on a timer) keeps
# the lifecycle predictable without a background task.
JANITOR_MAX_AGE_DAYS = 30

# Idempotent schema. ``CREATE TABLE IF NOT EXISTS`` makes connect()
# safe to call against a fresh DB or an existing one. Index on
# ``last_active_at`` makes a future janitor (DELETE WHERE
# last_active_at < ?) cheap; not on the hot read path (PK lookups).
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id      INTEGER PRIMARY KEY,
    session_id     TEXT    NOT NULL,
    repo_url       TEXT,
    ref            TEXT    NOT NULL DEFAULT 'HEAD',
    is_private     INTEGER NOT NULL DEFAULT 0,
    created_at     REAL    NOT NULL,
    last_active_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active
    ON sessions(last_active_at);
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class Session:
    """Represents a Claude Code session bound to a Discord thread.

    The workspace path is **derived**, not stored — it's always
    ``/vol/workspaces/<thread_id>``, set by the sandbox's
    ``provision_workspace`` Modal function. Storing it as a property
    here keeps the bot side from accidentally drifting from the
    sandbox's authoritative path layout.

    ``repo_url`` and ``ref`` capture the channel's repo binding at
    session creation time. Once a thread is started, its binding is
    grandfathered for the lifetime of the session — even if someone
    later ``/setrepo``s the channel to a different repo, in-flight
    threads keep their original repo. With persistence, this
    grandfathering also holds across bot restarts.
    """

    session_id: str
    thread_id: int
    repo_url: str | None = None
    ref: str = "HEAD"
    # Whether the bound repo was tagged private on the server's
    # allowlist at session-creation time. Drives:
    #   • Refuse-and-instruct on missing PAT in the sandbox-side
    #     ``run_claude_code`` (skip the provision call entirely
    #     and yield a clear error).
    #   • LiveStatus 🔒 prefix in the repo subtitle.
    # Always False for unbound channels and for sessions whose
    # repo binding predates the visibility marker (legacy entries
    # default to public on read).
    is_private: bool = False
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    @property
    def workspace_path(self) -> str:
        """Deterministic per-thread workspace path on the Modal volume.

        Mirrors ``WORKSPACES_ROOT`` in
        ``delulu_sandbox_modal.repo_provisioner``. Kept as a property
        rather than a stored field so the bot can never drift from
        the sandbox's source of truth — the sandbox's
        ``provision_workspace`` is what actually creates the
        directory; this property is just for logging and dispatch.
        """
        return f"/vol/workspaces/{self.thread_id}"

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.last_active_at) > ttl_seconds


class SessionManager:
    """SQLite-backed mapping of Discord thread → Claude Code session.

    Caller MUST ``await connect()`` before any other method, and
    SHOULD ``await close()`` at shutdown. ``connect`` is idempotent
    (re-calling it is a no-op once connected); ``close`` is too.
    Methods raise ``RuntimeError`` if used before ``connect``.

    Thread IDs are the primary key. When a thread's session expires,
    the row stays so ``get_or_create`` can carry forward the prior
    binding to the fresh session — same Discord thread, same repo,
    fresh Claude Code session. The workspace_path is deterministic
    per thread_id (a property on ``Session``), so this Just Works
    without any explicit reuse logic.
    """

    def __init__(self, db_path: str | Path, ttl_seconds: int = 3600) -> None:
        self._db_path = Path(db_path)
        self._ttl = ttl_seconds
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the SQLite connection and apply pragmas + schema.

        Idempotent. Sweeps rows older than ``JANITOR_MAX_AGE_DAYS``
        before serving any request, so the table size stays bounded
        without a background task.
        """
        if self._conn is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(self._db_path)
        try:
            # WAL: concurrent readers, atomic writer, robust to crash.
            # synchronous=NORMAL: durable enough for session state
            # (losing the last few writes on a power cut is fine).
            # busy_timeout: tolerate momentary writer-lock contention.
            await conn.executescript(
                "PRAGMA journal_mode=WAL;"
                "PRAGMA synchronous=NORMAL;"
                "PRAGMA foreign_keys=ON;"
                "PRAGMA busy_timeout=5000;" + _SCHEMA_SQL
            )
            # INSERT OR IGNORE so a reconnect on an existing DB
            # doesn't clobber a version a future migration bumped.
            await conn.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
                (SCHEMA_VERSION_KEY, SCHEMA_VERSION),
            )
            await conn.commit()
            self._conn = conn
            await self._run_janitor()
        except Exception:
            self._conn = None
            await conn.close()
            raise

        logger.info("session_manager.connected", db_path=str(self._db_path))

    async def _run_janitor(self) -> None:
        conn = self._require_conn()
        cutoff = time.time() - JANITOR_MAX_AGE_DAYS * 86400
        async with conn.execute(
            "DELETE FROM sessions WHERE last_active_at < ?",
            (cutoff,),
        ) as cursor:
            deleted = cursor.rowcount
        await conn.commit()
        if deleted:
            logger.info("session_manager.janitor_swept", deleted=deleted)

    async def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        if self._conn is None:
            return
        await self._conn.close()
        self._conn = None

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError(
                "SessionManager.connect() must be awaited before use",
            )
        return self._conn

    async def create_session(
        self,
        thread_id: int,
        *,
        repo_url: str | None = None,
        ref: str = "HEAD",
        is_private: bool = False,
    ) -> Session:
        """Create (or replace) the session row for ``thread_id``.

        ``repo_url`` and ``ref`` come from the channel's binding (via
        ``RepoConfig``) at thread-creation time — see
        ``MessageHandler.handle_channel_message``. ``is_private`` is
        sourced from the server's allowlist at the same moment.
        Caching all three on the row avoids per-reply RepoConfig +
        allowlist lookups and survives bot restarts.

        ``ON CONFLICT DO UPDATE`` is a deliberate replace — there is
        only ever one row per thread, and the only legitimate reason
        to call ``create_session`` on an existing thread is TTL
        expiry, where ``get_or_create`` has already read the prior
        row to inherit its bindings before replacing it.
        """
        conn = self._require_conn()
        session_id = uuid.uuid4().hex[:12]
        now = time.time()
        await conn.execute(
            """
            INSERT INTO sessions (
                thread_id, session_id, repo_url, ref, is_private,
                created_at, last_active_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                session_id     = excluded.session_id,
                repo_url       = excluded.repo_url,
                ref            = excluded.ref,
                is_private     = excluded.is_private,
                created_at     = excluded.created_at,
                last_active_at = excluded.last_active_at
            """,
            (thread_id, session_id, repo_url, ref, int(is_private), now, now),
        )
        await conn.commit()

        session = Session(
            session_id=session_id,
            thread_id=thread_id,
            repo_url=repo_url,
            ref=ref,
            is_private=is_private,
            created_at=now,
            last_active_at=now,
        )
        logger.info(
            "session.created",
            session_id=session_id,
            thread_id=thread_id,
            workspace_path=session.workspace_path,
            repo_url=repo_url,
            ref=ref,
            is_private=is_private,
        )
        return session

    async def get_session(self, thread_id: int) -> Session | None:
        """Return the session for ``thread_id`` if present and not expired.

        On hit, bumps ``last_active_at`` to now — but only persists
        the bump when the row hasn't been touched within
        ``TOUCH_DEBOUNCE_SECONDS``. The in-memory copy is always
        updated so TTL gating stays exact within a process; the DB
        write is what we're throttling, since this is on the
        every-message hot path.
        """
        row = await self._read_row(thread_id)
        if row is None:
            return None
        session = _row_to_session(row)
        if session.is_expired(self._ttl):
            logger.info(
                "session.expired",
                session_id=session.session_id,
                thread_id=thread_id,
            )
            return None

        now = time.time()
        if now - session.last_active_at >= TOUCH_DEBOUNCE_SECONDS:
            conn = self._require_conn()
            await conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
            await conn.commit()
        session.last_active_at = now
        return session

    async def get_or_create(self, thread_id: int) -> tuple[Session, bool]:
        """Return ``(session, is_new)`` — fresh session inherits prior binding.

        On TTL expiry the prior row's ``repo_url`` / ``ref`` /
        ``is_private`` are carried forward to the fresh session.
        This is what makes a long-running thread keep working after
        the bot's session TTL fires; the user shouldn't have to
        re-bind a repo just because their thread went idle for an
        hour. With persistence, the same inheritance now also
        survives a bot restart.
        """
        existing = await self.get_session(thread_id)
        if existing is not None:
            return existing, False

        # Read the raw row (bypasses TTL gating in get_session) so we
        # can inherit prior bindings even after expiry.
        old_row = await self._read_row(thread_id)
        if old_row is not None:
            old = _row_to_session(old_row)
            session = await self.create_session(
                thread_id,
                repo_url=old.repo_url,
                ref=old.ref,
                is_private=old.is_private,
            )
            logger.info(
                "session.reused_workspace",
                old_session=old.session_id,
                new_session=session.session_id,
                workspace_path=session.workspace_path,
                repo_url=session.repo_url,
                ref=session.ref,
                is_private=session.is_private,
            )
        else:
            session = await self.create_session(thread_id)

        return session, True

    async def remove(self, thread_id: int) -> None:
        """Delete a thread's session row. No-op if not present."""
        conn = self._require_conn()
        await conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        await conn.commit()

    async def active_count(self) -> int:
        """Count sessions whose last_active_at is within TTL.

        For monitoring/logging only — fine if it's a few ms stale.
        """
        conn = self._require_conn()
        cutoff = time.time() - self._ttl
        async with conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE last_active_at >= ?",
            (cutoff,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def _read_row(self, thread_id: int) -> tuple | None:
        conn = self._require_conn()
        async with conn.execute(
            """
            SELECT thread_id, session_id, repo_url, ref, is_private,
                   created_at, last_active_at
            FROM sessions WHERE thread_id = ?
            """,
            (thread_id,),
        ) as cursor:
            return await cursor.fetchone()


def _row_to_session(row: tuple) -> Session:
    """Materialize a Session dataclass from a sessions-table row."""
    return Session(
        thread_id=row[0],
        session_id=row[1],
        repo_url=row[2],
        ref=row[3],
        is_private=bool(row[4]),
        created_at=row[5],
        last_active_at=row[6],
    )
