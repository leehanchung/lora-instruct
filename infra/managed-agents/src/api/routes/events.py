"""Event sending and SSE streaming endpoints."""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db, get_redis
from src.api.models import (
    AgentMessageEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
    EventSend,
    ListResponse,
    SessionStatusIdleEvent,
)

router = APIRouter(prefix="/v1/sessions", tags=["events"])

# TODO: Replace with actual database once schema is defined
_events_store = {}  # In-memory event store: session_id -> list of events


def _get_redis_channel(session_id: str) -> str:
    """Get Redis pub/sub channel name for a session."""
    return f"session:{session_id}:events"


@router.post("/{session_id}/events", status_code=status.HTTP_202_ACCEPTED)
async def send_events(
    session_id: str,
    req: EventSend,
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
) -> dict:
    """Send events to a session.

    Accepts user messages, interrupts, and custom tool results.
    Publishes to Redis for real-time streaming.
    """
    # Validate session exists
    # TODO: Query from database
    if session_id not in _events_store:
        _events_store[session_id] = []

    # Process and publish each event
    channel = _get_redis_channel(session_id)
    event_count = 0

    for event in req.events:
        event_dict = event.model_dump(mode="json")
        event_json = json.dumps(event_dict)

        # Publish to Redis for streaming clients
        # TODO: Implement actual redis.publish(channel, event_json)
        # await redis.publish(channel, event_json)

        # Store in database
        # TODO: Store in database via ORM
        _events_store[session_id].append(event_dict)
        event_count += 1

    return {
        "session_id": session_id,
        "events_accepted": event_count,
    }


@router.get("/{session_id}/events", response_model=ListResponse)
async def list_events(
    session_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> ListResponse:
    """List events for a session with pagination."""
    # TODO: Query from database
    if session_id not in _events_store:
        return ListResponse(data=[], next_page=None)

    events = _events_store[session_id][offset : offset + limit]

    return ListResponse(
        data=events,
        next_page=None,  # TODO: Calculate based on offset/limit
    )


async def _event_stream_generator(
    session_id: str,
    redis,
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for a session's events.

    Reads from Redis pub/sub channel and yields formatted SSE messages.
    """
    # TODO: Implement actual Redis pub/sub subscription
    # redis_pubsub = redis.pubsub()
    # channel = _get_redis_channel(session_id)
    # await redis_pubsub.subscribe(channel)

    # Initial connection heartbeat
    yield ":connected\n\n"

    # TODO: Listen for messages on channel and yield them
    # while True:
    #     message = await redis_pubsub.get_message(ignore_subscribe_messages=True)
    #     if message:
    #         event_json = message["data"].decode()
    #         yield f"data: {event_json}\n\n"
    #     else:
    #         await asyncio.sleep(0.1)  # Small delay to avoid busy waiting

    # For now, yield a placeholder
    yield "data: {\"type\": \"placeholder\"}\n\n"


@router.get("/{session_id}/stream")
async def stream_events(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
) -> StreamingResponse:
    """Stream events for a session via Server-Sent Events (SSE).

    Establishes long-lived connection that receives events in real-time.
    """
    # TODO: Validate session exists
    # if session_id not in _sessions_store:
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail=f"Session {session_id} not found",
    #     )

    return StreamingResponse(
        _event_stream_generator(session_id, redis),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
