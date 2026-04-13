"""SQLAlchemy async models for managed agents."""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Index, JSON, String, Text, DateTime, func
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class SessionStatus(str, Enum):
    """Session status enumeration."""

    RESCHEDULING = "rescheduling"
    RUNNING = "running"
    IDLE = "idle"
    TERMINATED = "terminated"


class Agent(Base):
    """Agent configuration and metadata."""

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    tools: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    mcp_servers: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    version: Mapped[int] = mapped_column(default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Agent(id={self.id}, name={self.name}, version={self.version})>"


class Environment(Base):
    """Execution environment configuration."""

    __tablename__ = "environments"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    archived_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<Environment(id={self.id}, name={self.name})>"


class Session(Base):
    """Agent session."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    agent_snapshot: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )
    environment_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    resources: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[SessionStatus] = mapped_column(
        default=SessionStatus.RESCHEDULING, nullable=False
    )
    stats: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    usage: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    archived_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_session_agent_id", "agent_id"),
        Index("ix_session_environment_id", "environment_id"),
        Index("ix_session_status", "status"),
        Index("ix_session_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, agent_id={self.agent_id}, status={self.status})>"


class Event(Base):
    """Session event log."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_event_session_id_created_at", "session_id", "created_at"),
        Index("ix_event_type", "type"),
        Index("ix_event_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Event(id={self.id}, session_id={self.session_id}, type={self.type})>"


__all__ = [
    "Base",
    "Agent",
    "Environment",
    "Session",
    "Event",
    "SessionStatus",
]
