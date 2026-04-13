"""Database session management."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker

from src.db.models import Base

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/managed_agents",
)

# Connection pool configuration
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# Global engine and session maker
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[sessionmaker] = None


async def init_db() -> None:
    """Initialize database engine and create all tables.

    Creates the async engine and sessionmaker, then creates all tables.
    Safe to call multiple times.

    Raises:
        Exception: If engine creation or table creation fails
    """
    global _engine, _async_session_maker

    if _engine is not None:
        logger.info("Database already initialized")
        return

    try:
        _engine = create_async_engine(
            DATABASE_URL,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true",
            pool_size=POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_recycle=POOL_RECYCLE,
            pool_pre_ping=True,
        )

        _async_session_maker = sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

        # Create tables
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session.

    Yields async session for use in FastAPI dependencies or context managers.
    Ensures proper cleanup on completion or error.

    Usage:
        async with get_db_session() as session:
            result = await session.execute(query)

    Yields:
        AsyncSession: Database session

    Raises:
        RuntimeError: If database not initialized
    """
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions.

    Useful for direct usage without FastAPI dependency injection.

    Usage:
        async with get_db_context() as session:
            result = await session.execute(query)

    Yields:
        AsyncSession: Database session
    """
    async for session in get_db_session():
        yield session


async def close_db() -> None:
    """Close database connections.

    Closes the engine and cleans up connection pool.
    Should be called during application shutdown.
    """
    global _engine, _async_session_maker

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
        logger.info("Database connections closed")


async def health_check() -> bool:
    """Perform a database health check.

    Attempts a simple query to verify connectivity.

    Returns:
        True if database is healthy, False otherwise
    """
    if _engine is None:
        return False

    try:
        async with _async_session_maker() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


__all__ = [
    "DATABASE_URL",
    "POOL_SIZE",
    "MAX_OVERFLOW",
    "POOL_RECYCLE",
    "init_db",
    "get_db_session",
    "get_db_context",
    "close_db",
    "health_check",
]
