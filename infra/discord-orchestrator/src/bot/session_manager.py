"""Maps Discord threads to Claude Code sessions with TTL expiry."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class Session:
    """Represents a Claude Code session bound to a Discord thread."""

    session_id: str
    thread_id: int
    workspace_path: str
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.last_active_at) > ttl_seconds

    def touch(self) -> None:
        self.last_active_at = time.time()


class SessionManager:
    """Thread-to-session mapping with TTL-based expiry.

    Thread IDs are the primary key. When a thread's session expires,
    we keep the workspace but start a fresh Claude Code session.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._sessions: dict[int, Session] = {}
        self._ttl = ttl_seconds

    def create_session(self, thread_id: int, repo_url: str | None = None) -> Session:
        """Create a new session for a thread."""
        session_id = uuid.uuid4().hex[:12]
        workspace_path = f"/vol/workspaces/{session_id}"

        session = Session(
            session_id=session_id,
            thread_id=thread_id,
            workspace_path=workspace_path,
        )
        self._sessions[thread_id] = session

        logger.info(
            "session.created",
            session_id=session_id,
            thread_id=thread_id,
            workspace_path=workspace_path,
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
        """Get existing session or create new one. Returns (session, is_new)."""
        existing = self.get_session(thread_id)
        if existing is not None:
            return existing, False

        # Check if there was a previous (expired) session — reuse workspace path
        old = self._sessions.get(thread_id)
        session = self.create_session(thread_id)

        if old is not None:
            # Reuse the workspace directory from the expired session
            session.workspace_path = old.workspace_path
            logger.info(
                "session.reused_workspace",
                old_session=old.session_id,
                new_session=session.session_id,
                workspace_path=session.workspace_path,
            )

        return session, True

    def remove(self, thread_id: int) -> None:
        """Remove a session mapping."""
        self._sessions.pop(thread_id, None)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if not s.is_expired(self._ttl))
