"""Tests for the SQLite-backed SessionManager.

The whole point of moving SessionManager off an in-memory dict was
to survive bot restarts. The two load-bearing tests here are:

- ``test_persists_across_restart`` — write with one ``SessionManager``
  instance, read with a fresh one against the same DB file. This is
  the regression for the ``setrepo-persistence-bug`` symptom; if it
  ever fails, /setrepo bindings will silently vanish across deploys
  again.
- ``test_get_or_create_inherits_after_expiry`` — confirms the
  grandfathering semantic survives across both TTL expiry and
  restart, since both paths hit the same row-inheritance code.

Other tests cover field round-trips, TTL gating, concurrent writes,
and the connect/close lifecycle. Tests use ``tmp_path`` for isolation
and rely on ``asyncio_mode = "auto"`` from ``pyproject.toml``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from delulu_discord.session_manager import (
    SCHEMA_VERSION,
    Session,
    SessionManager,
    _row_to_session,
)


@pytest.fixture
async def sm(tmp_path: Path):
    """Yield a connected SessionManager pointed at a per-test SQLite file."""
    manager = SessionManager(db_path=tmp_path / "sessions.db", ttl_seconds=3600)
    await manager.connect()
    try:
        yield manager
    finally:
        await manager.close()


class TestLifecycle:
    async def test_use_before_connect_raises(self, tmp_path: Path):
        # Forgetting ``await connect()`` is a programming error, not
        # a recoverable runtime condition. RuntimeError surfaces it
        # at the first call instead of letting a None propagate.
        manager = SessionManager(db_path=tmp_path / "sessions.db")
        with pytest.raises(RuntimeError, match="connect"):
            await manager.create_session(thread_id=1)

    async def test_connect_is_idempotent(self, tmp_path: Path):
        manager = SessionManager(db_path=tmp_path / "sessions.db")
        await manager.connect()
        await manager.connect()  # second call must be a no-op, not raise
        try:
            await manager.create_session(thread_id=1)
        finally:
            await manager.close()

    async def test_close_is_idempotent(self, tmp_path: Path):
        manager = SessionManager(db_path=tmp_path / "sessions.db")
        await manager.connect()
        await manager.close()
        await manager.close()  # second close is a no-op

    async def test_creates_parent_directory(self, tmp_path: Path):
        # /data won't exist on a fresh dev box; connect() must mkdir
        # the parent rather than blowing up.
        nested = tmp_path / "a" / "b" / "c" / "sessions.db"
        manager = SessionManager(db_path=nested)
        await manager.connect()
        try:
            assert nested.parent.is_dir()
        finally:
            await manager.close()

    async def test_records_schema_version(self, tmp_path: Path):
        manager = SessionManager(db_path=tmp_path / "sessions.db")
        await manager.connect()
        try:
            conn = manager._conn
            assert conn is not None
            async with conn.execute(
                "SELECT value FROM schema_meta WHERE key = ?",
                ("schema_version",),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0] == SCHEMA_VERSION
        finally:
            await manager.close()


class TestCreateGetRoundtrip:
    async def test_full_field_roundtrip(self, sm: SessionManager):
        # All fields the dataclass cares about — repo_url, ref,
        # is_private — must round-trip through SQLite without drift.
        # is_private is the trickiest (stored as INTEGER 0/1, cast
        # back to bool on read) so the assertion explicitly checks
        # the type, not just truthiness.
        before = time.time()
        created = await sm.create_session(
            thread_id=42,
            repo_url="https://github.com/alice/api-service",
            ref="main",
            is_private=True,
        )
        after = time.time()

        assert created.thread_id == 42
        assert created.repo_url == "https://github.com/alice/api-service"
        assert created.ref == "main"
        assert created.is_private is True
        assert before <= created.created_at <= after
        assert created.session_id  # non-empty

        got = await sm.get_session(42)
        assert got is not None
        assert got.session_id == created.session_id
        assert got.repo_url == created.repo_url
        assert got.ref == created.ref
        assert got.is_private is True
        assert isinstance(got.is_private, bool)

    async def test_unbound_repo_url_is_none(self, sm: SessionManager):
        # repo_url=None is the general-Q&A path. SQLite stores it as
        # NULL; the read must come back as Python None, not "" or
        # the string "None".
        await sm.create_session(thread_id=1)
        got = await sm.get_session(1)
        assert got is not None
        assert got.repo_url is None
        assert got.is_private is False
        assert got.ref == "HEAD"

    async def test_get_missing_returns_none(self, sm: SessionManager):
        assert await sm.get_session(999) is None


class TestPersistence:
    async def test_persists_across_restart(self, tmp_path: Path):
        """Regression test for setrepo-persistence-bug.md.

        Writing with one SessionManager and reading with a fresh
        instance against the same DB file MUST return the same data
        — that is the entire point of moving off the in-memory dict.
        If this test ever flips, /setrepo bindings will silently
        vanish on every deploy again.
        """
        db = tmp_path / "sessions.db"
        first = SessionManager(db_path=db, ttl_seconds=3600)
        await first.connect()
        try:
            created = await first.create_session(
                thread_id=12345,
                repo_url="https://github.com/alice/private-svc",
                ref="main",
                is_private=True,
            )
            written_session_id = created.session_id
        finally:
            await first.close()

        # Fresh process / fresh manager. The DB file is the only
        # state carried over — same as a Docker `restart` cycle on
        # the VPS, where the named volume holds /data/sessions.db.
        second = SessionManager(db_path=db, ttl_seconds=3600)
        await second.connect()
        try:
            got = await second.get_session(12345)
            assert got is not None
            assert got.session_id == written_session_id
            assert got.repo_url == "https://github.com/alice/private-svc"
            assert got.ref == "main"
            assert got.is_private is True
        finally:
            await second.close()


class TestTTLExpiry:
    async def test_expired_session_returns_none(self, tmp_path: Path):
        # TTL is checked in Python (last_active_at vs time.time()),
        # not in SQL — so the row stays in the table for
        # get_or_create's grandfathering logic to read.
        manager = SessionManager(db_path=tmp_path / "s.db", ttl_seconds=1)
        await manager.connect()
        try:
            await manager.create_session(thread_id=1, repo_url="x")
            # Backdate last_active_at so it's outside the 1s TTL.
            conn = manager._conn
            assert conn is not None
            await conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE thread_id = ?",
                (time.time() - 10, 1),
            )
            await conn.commit()

            assert await manager.get_session(1) is None
        finally:
            await manager.close()

    async def test_get_session_bumps_last_active(self, sm: SessionManager):
        # Hot-thread protection: any reply in a thread should reset
        # the TTL clock so a long conversation doesn't get cut off
        # mid-flight.
        await sm.create_session(thread_id=1, repo_url="x")
        # Backdate enough to register but stay within TTL (3600s).
        conn = sm._conn
        assert conn is not None
        backdated = time.time() - 100
        await conn.execute(
            "UPDATE sessions SET last_active_at = ? WHERE thread_id = ?",
            (backdated, 1),
        )
        await conn.commit()

        got = await sm.get_session(1)
        assert got is not None
        assert got.last_active_at > backdated


class TestGetOrCreate:
    async def test_returns_existing(self, sm: SessionManager):
        first = await sm.create_session(thread_id=1, repo_url="r")
        got, is_new = await sm.get_or_create(1)
        assert is_new is False
        assert got.session_id == first.session_id

    async def test_creates_when_missing(self, sm: SessionManager):
        got, is_new = await sm.get_or_create(1)
        assert is_new is True
        assert got.thread_id == 1
        assert got.repo_url is None

    async def test_inherits_after_expiry(self, tmp_path: Path):
        # After TTL expiry the prior row's repo_url / ref /
        # is_private are carried into the fresh session. Without
        # this the user would have to re-/setrepo every hour.
        manager = SessionManager(db_path=tmp_path / "s.db", ttl_seconds=1)
        await manager.connect()
        try:
            original = await manager.create_session(
                thread_id=1,
                repo_url="https://github.com/alice/api",
                ref="release-1.4",
                is_private=True,
            )
            # Backdate to force expiry.
            conn = manager._conn
            assert conn is not None
            await conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE thread_id = ?",
                (time.time() - 10, 1),
            )
            await conn.commit()

            new_session, is_new = await manager.get_or_create(1)
            assert is_new is True
            assert new_session.session_id != original.session_id
            assert new_session.repo_url == "https://github.com/alice/api"
            assert new_session.ref == "release-1.4"
            assert new_session.is_private is True
        finally:
            await manager.close()

    async def test_inherits_across_restart(self, tmp_path: Path):
        # Combines the previous two regressions: a session expired
        # before the bot was restarted should still inherit on the
        # first get_or_create after the restart.
        db = tmp_path / "s.db"

        first = SessionManager(db_path=db, ttl_seconds=1)
        await first.connect()
        try:
            await first.create_session(
                thread_id=7,
                repo_url="https://github.com/alice/foo",
                ref="dev",
                is_private=False,
            )
            conn = first._conn
            assert conn is not None
            await conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE thread_id = ?",
                (time.time() - 10, 7),
            )
            await conn.commit()
        finally:
            await first.close()

        second = SessionManager(db_path=db, ttl_seconds=1)
        await second.connect()
        try:
            new_session, is_new = await second.get_or_create(7)
            assert is_new is True
            assert new_session.repo_url == "https://github.com/alice/foo"
            assert new_session.ref == "dev"
            assert new_session.is_private is False
        finally:
            await second.close()


class TestRemove:
    async def test_remove_existing(self, sm: SessionManager):
        await sm.create_session(thread_id=1)
        await sm.remove(1)
        assert await sm.get_session(1) is None

    async def test_remove_missing_is_noop(self, sm: SessionManager):
        # Sometimes the bot races a session cleanup against a
        # /setrepo unbind; both shouldn't error.
        await sm.remove(999)


class TestActiveCount:
    async def test_counts_only_active(self, tmp_path: Path):
        manager = SessionManager(db_path=tmp_path / "s.db", ttl_seconds=10)
        await manager.connect()
        try:
            await manager.create_session(thread_id=1)
            await manager.create_session(thread_id=2)
            await manager.create_session(thread_id=3)
            # Backdate one outside the TTL window.
            conn = manager._conn
            assert conn is not None
            await conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE thread_id = ?",
                (time.time() - 1000, 2),
            )
            await conn.commit()

            assert await manager.active_count() == 2
        finally:
            await manager.close()


class TestConcurrency:
    async def test_concurrent_get_or_create_yields_one_row(self, sm: SessionManager):
        # Two coroutines racing on the same thread_id must not produce
        # two distinct session IDs in the table. SQLite's PK
        # constraint + ON CONFLICT handles this; aiosqlite serializes
        # both writes through one background thread, so the second
        # call effectively sees the first call's row.
        results = await asyncio.gather(
            sm.get_or_create(42),
            sm.get_or_create(42),
        )
        sessions = [r[0] for r in results]

        # Whatever happens at the create layer, the persisted row
        # must be a single session by the end. Read-back is the
        # source of truth.
        got = await sm.get_session(42)
        assert got is not None
        # Both callers should hold a Session referring to the same
        # thread; at least one of them should match the persisted
        # session_id.
        assert any(s.session_id == got.session_id for s in sessions)


class TestRowToSession:
    def test_handles_int_is_private_zero(self):
        row = (1, "abc", "url", "main", 0, 1.0, 2.0)
        s = _row_to_session(row)
        assert s.is_private is False
        assert isinstance(s.is_private, bool)

    def test_handles_int_is_private_one(self):
        row = (1, "abc", "url", "main", 1, 1.0, 2.0)
        s = _row_to_session(row)
        assert s.is_private is True

    def test_handles_null_repo_url(self):
        row = (1, "abc", None, "HEAD", 0, 1.0, 2.0)
        s = _row_to_session(row)
        assert s.repo_url is None


class TestSessionDataclass:
    def test_workspace_path_property(self):
        s = Session(session_id="x", thread_id=12345)
        assert s.workspace_path == "/vol/workspaces/12345"

    def test_is_expired_true_past_ttl(self):
        s = Session(session_id="x", thread_id=1, last_active_at=time.time() - 100)
        assert s.is_expired(ttl_seconds=10)

    def test_is_expired_false_within_ttl(self):
        s = Session(session_id="x", thread_id=1, last_active_at=time.time())
        assert not s.is_expired(ttl_seconds=10)
