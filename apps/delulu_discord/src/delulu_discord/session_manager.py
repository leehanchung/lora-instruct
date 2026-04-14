"""Maps Discord threads to Claude Code sessions with TTL expiry."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


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
    threads keep their original repo. This matches the PRD's
    "thread binding is captured at thread creation" decision.
    """

    session_id: str
    thread_id: int
    repo_url: str | None = None
    ref: str = "HEAD"
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

    def touch(self) -> None:
        self.last_active_at = time.time()


class SessionManager:
    """Thread-to-session mapping with TTL-based expiry.

    Thread IDs are the primary key. When a thread's session expires,
    we keep the workspace but start a fresh Claude Code session — the
    workspace_path is deterministic per thread_id so this Just Works
    without any explicit reuse logic.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._sessions: dict[int, Session] = {}
        self._ttl = ttl_seconds

    def create_session(
        self,
        thread_id: int,
        *,
        repo_url: str | None = None,
        ref: str = "HEAD",
    ) -> Session:
        """Create a new session for a thread, optionally bound to a repo.

        ``repo_url`` and ``ref`` come from the channel's binding (via
        ``RepoConfig``) at thread-creation time — see
        ``MessageHandler.handle_channel_message``. Stored on the
        Session so subsequent thread replies inherit the binding
        without re-querying ``RepoConfig`` and without re-checking
        the allowlist (bindings are grandfathered).

        The workspace path is deterministic per thread_id — same
        directory on the Modal volume across bot restarts and TTL
        expiries — so Claude Code's ``--continue`` (which keys its
        history off cwd) finds the prior conversation.
        """
        session_id = uuid.uuid4().hex[:12]

        session = Session(
            session_id=session_id,
            thread_id=thread_id,
            repo_url=repo_url,
            ref=ref,
        )
        self._sessions[thread_id] = session

        logger.info(
            "session.created",
            session_id=session_id,
            thread_id=thread_id,
            workspace_path=session.workspace_path,
            repo_url=repo_url,
            ref=ref,
        )
        return session

    def get_session(self, thread_id: int) -> Session | None:
        """Get session for a thread, or None if not found / expired."""
        session = self._sessions.get(thread_id)
        if session is None:
            return None

        if session.is_expired(self._ttl):
            logger.info(
                "session.expired",
                session_id=session.session_id,
                thread_id=thread_id,
            )
            # Keep workspace, but create fresh session
            return None

        session.touch()
        return session

    def get_or_create(self, thread_id: int) -> tuple[Session, bool]:
        """Get existing session or create new one. Returns (session, is_new).

        On TTL expiry the prior session's ``repo_url`` / ``ref`` are
        carried over to the fresh session — same Discord thread, same
        repo binding, fresh Claude Code session. This is what makes
        a long-running thread keep working after the bot's session
        TTL fires; the user shouldn't have to re-bind a repo just
        because their thread went idle for an hour.
        """
        existing = self.get_session(thread_id)
        if existing is not None:
            return existing, False

        # Check if there was a previous (expired) session and inherit
        # its repo binding so the fresh session keeps targeting the
        # same repo.
        old = self._sessions.get(thread_id)
        if old is not None:
            session = self.create_session(
                thread_id,
                repo_url=old.repo_url,
                ref=old.ref,
            )
            logger.info(
                "session.reused_workspace",
                old_session=old.session_id,
                new_session=session.session_id,
                workspace_path=session.workspace_path,
                repo_url=session.repo_url,
                ref=session.ref,
            )
        else:
            session = self.create_session(thread_id)

        return session, True

    def remove(self, thread_id: int) -> None:
        """Remove a session mapping."""
        self._sessions.pop(thread_id, None)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if not s.is_expired(self._ttl))
