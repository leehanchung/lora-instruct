"""Redis event publisher for agent runtime events."""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event structure."""

    event_id: str
    timestamp: str
    event_type: str
    session_id: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)


class EventPublisher:
    """Publishes agent runtime events to Redis pub/sub."""

    def __init__(self, session_id: str, redis_url: str = "redis://localhost:6379"):
        """Initialize event publisher.

        Args:
            session_id: Unique session identifier
            redis_url: Redis connection URL
        """
        self.session_id = session_id
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.channel = f"session:{session_id}:events"

    async def connect(self) -> None:
        """Establish connection to Redis."""
        self.redis_client = await redis.from_url(
            self.redis_url, decode_responses=True, socket_connect_timeout=5
        )
        logger.info(f"Connected to Redis at {self.redis_url}")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")

    async def _publish(self, event: Event) -> None:
        """Publish event to Redis channel.

        Args:
            event: Event to publish
        """
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        event_json = json.dumps(event.to_dict())
        await self.redis_client.publish(self.channel, event_json)
        logger.debug(f"Published event {event.event_type}: {event.event_id}")

    async def publish_agent_message(self, content: str) -> str:
        """Publish agent text message.

        Args:
            content: Message content

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = Event(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="agent.message",
            session_id=self.session_id,
            payload={"content": content, "type": "text"},
        )
        await self._publish(event)
        return event_id

    async def publish_tool_use(
        self, tool_name: str, tool_id: str, tool_input: dict[str, Any]
    ) -> str:
        """Publish tool use event.

        Args:
            tool_name: Name of tool being used
            tool_id: Unique ID for this tool call
            tool_input: Input parameters to tool

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = Event(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="agent.tool_use",
            session_id=self.session_id,
            payload={
                "tool_id": tool_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
        )
        await self._publish(event)
        return event_id

    async def publish_tool_result(
        self,
        tool_id: str,
        tool_name: str,
        result: str,
        error: Optional[str] = None,
    ) -> str:
        """Publish tool result event.

        Args:
            tool_id: ID of tool call this result corresponds to
            tool_name: Name of tool that was called
            result: Result content
            error: Optional error message if tool failed

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "result": result,
        }
        if error:
            payload["error"] = error

        event = Event(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="agent.tool_result",
            session_id=self.session_id,
            payload=payload,
        )
        await self._publish(event)
        return event_id

    async def publish_status_idle(self) -> str:
        """Publish session idle status (agent done processing).

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = Event(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="session.status_idle",
            session_id=self.session_id,
            payload={},
        )
        await self._publish(event)
        return event_id

    async def publish_custom_tool_use(self, tool_id: str, tool_input: dict[str, Any]) -> str:
        """Publish custom tool use event (waiting for external result).

        Args:
            tool_id: Unique ID for this custom tool call
            tool_input: Input parameters for custom tool

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = Event(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="agent.custom_tool_use",
            session_id=self.session_id,
            payload={"tool_id": tool_id, "tool_input": tool_input},
        )
        await self._publish(event)
        return event_id

    async def wait_for_custom_tool_result(
        self, tool_id: str, timeout: int = 300
    ) -> dict[str, Any]:
        """Wait for custom tool result from Redis.

        Args:
            tool_id: ID of custom tool call
            timeout: Max seconds to wait

        Returns:
            Tool result payload

        Raises:
            TimeoutError: If result not received within timeout
        """
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")

        result_channel = f"session:{self.session_id}:custom_tool_result:{tool_id}"
        pubsub = self.redis_client.pubsub()

        try:
            await pubsub.subscribe(result_channel)
            logger.debug(f"Waiting for custom tool result on {result_channel}")

            # Wait for message with timeout
            start_time = asyncio.get_event_loop().time()
            while True:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0,
                )

                if message:
                    result_data = json.loads(message["data"])
                    logger.debug(f"Received custom tool result: {tool_id}")
                    return result_data

                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Custom tool result not received within {timeout}s"
                    )

        finally:
            await pubsub.unsubscribe(result_channel)
            await pubsub.close()
