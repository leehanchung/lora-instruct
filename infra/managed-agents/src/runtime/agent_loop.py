"""Core agent runtime loop."""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

import redis.asyncio as redis

from .llm.base import LLMClient, create_llm_client
from .streaming import EventPublisher
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    """Core agent autonomous loop."""

    def __init__(
        self,
        session_id: str,
        agent_config: dict[str, Any],
        llm_client: LLMClient,
        event_publisher: EventPublisher,
        tool_registry: ToolRegistry,
        max_turns: int = 50,
    ):
        """Initialize agent loop.

        Args:
            session_id: Unique session identifier
            agent_config: Agent configuration (model, system prompt, tools)
            llm_client: LLM client for completions
            event_publisher: Event publisher for Redis
            tool_registry: Registry of available tools
            max_turns: Max turns before stopping (default 50)
        """
        self.session_id = session_id
        self.agent_config = agent_config
        self.llm_client = llm_client
        self.event_publisher = event_publisher
        self.tool_registry = tool_registry
        self.max_turns = max_turns

        self.system_prompt = agent_config.get("system_prompt", "")
        self.should_stop = False

        logger.info(f"Initialized AgentLoop for session {session_id}")

    def handle_interrupt(self) -> None:
        """Signal to stop the loop after current tool execution."""
        self.should_stop = True
        logger.info("Agent loop interrupt requested")

    async def run(self, initial_message: str) -> None:
        """Run the agent loop.

        Args:
            initial_message: Initial user message to process
        """
        logger.info(f"Starting agent loop with message: {initial_message[:100]}")

        # Initialize message history
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": initial_message},
        ]

        turn = 0
        while turn < self.max_turns:
            if self.should_stop:
                logger.info("Agent loop interrupted, stopping")
                break

            turn += 1
            logger.debug(f"Agent turn {turn}/{self.max_turns}")

            try:
                # Call LLM
                response = await self.llm_client.complete(
                    messages=messages,
                    tools=self.tool_registry.get_tool_definitions(),
                    system=self.system_prompt,
                    max_tokens=4096,
                )

                # Process response content
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": [],
                }

                has_tool_use = False
                has_text = False

                for block in response.content:
                    if block.type == "text":
                        # Text response
                        await self.event_publisher.publish_agent_message(block.text)
                        assistant_message["content"].append(
                            {
                                "type": "text",
                                "text": block.text,
                            }
                        )
                        has_text = True

                    elif block.type == "tool_use":
                        # Tool use
                        has_tool_use = True
                        tool_id = block.id
                        tool_name = block.name
                        tool_input = block.input

                        logger.debug(f"Agent using tool: {tool_name}")
                        await self.event_publisher.publish_tool_use(
                            tool_name=tool_name,
                            tool_id=tool_id,
                            tool_input=tool_input,
                        )

                        assistant_message["content"].append(
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool_name,
                                "input": tool_input,
                            }
                        )

                        # Check if custom tool
                        if tool_name.startswith("custom:"):
                            # Wait for external result
                            logger.info(f"Waiting for custom tool result: {tool_name}")
                            await self.event_publisher.publish_custom_tool_use(
                                tool_id=tool_id,
                                tool_input=tool_input,
                            )

                            try:
                                result_data = await self.event_publisher.wait_for_custom_tool_result(
                                    tool_id=tool_id,
                                    timeout=300,
                                )
                                tool_result = result_data.get("result", "")
                                tool_error = result_data.get("error")
                            except TimeoutError as e:
                                logger.error(f"Custom tool timeout: {tool_name}")
                                tool_result = ""
                                tool_error = str(e)
                        else:
                            # Execute built-in tool
                            exec_result = await self.tool_registry.execute(
                                tool_name=tool_name,
                                tool_input=tool_input,
                            )

                            tool_result = exec_result.output
                            tool_error = exec_result.error if exec_result.exit_code != 0 else None

                        # Publish result
                        await self.event_publisher.publish_tool_result(
                            tool_id=tool_id,
                            tool_name=tool_name,
                            result=tool_result,
                            error=tool_error,
                        )

                        # Add to message history for next turn
                        messages.append(assistant_message)
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": tool_result or tool_error or "No output",
                                    }
                                ],
                            }
                        )

                # Check if we should continue
                if not has_tool_use:
                    # No more tool calls, agent is done
                    messages.append(assistant_message)
                    logger.info("Agent completed (no more tool calls)")
                    break

                # Continue loop for next turn

            except Exception as e:
                logger.error(f"Error in agent loop turn {turn}: {e}", exc_info=True)
                await self.event_publisher.publish_tool_result(
                    tool_id=str(uuid.uuid4()),
                    tool_name="error",
                    result="",
                    error=f"Agent error: {str(e)}",
                )
                break

        # Signal completion
        await self.event_publisher.publish_status_idle()
        logger.info(f"Agent loop completed after {turn} turns")


async def main() -> None:
    """Entry point for agent runtime.

    Reads configuration from environment variables:
    - SESSION_ID: Unique session ID
    - AGENT_CONFIG_JSON: Agent config as JSON
    - REDIS_URL: Redis connection URL
    - INITIAL_MESSAGE: Initial message to process
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Read environment variables
    session_id = os.getenv("SESSION_ID")
    agent_config_json = os.getenv("AGENT_CONFIG_JSON", "{}")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    initial_message = os.getenv("INITIAL_MESSAGE", "Hello")

    if not session_id:
        raise ValueError("SESSION_ID environment variable is required")

    # Parse config
    try:
        agent_config = json.loads(agent_config_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AGENT_CONFIG_JSON: {e}")
        raise

    logger.info(f"Starting agent runtime for session {session_id}")
    logger.info(f"LLM config: {agent_config.get('llm', {})}")

    # Initialize components
    event_publisher = EventPublisher(session_id=session_id, redis_url=redis_url)
    await event_publisher.connect()

    try:
        # Create LLM client
        llm_config = agent_config.get("llm", {})
        llm_client = create_llm_client(llm_config)

        # Setup tool registry
        tool_registry = ToolRegistry()
        tool_registry.register_builtin_tools()

        # Create and run agent loop
        agent_loop = AgentLoop(
            session_id=session_id,
            agent_config=agent_config,
            llm_client=llm_client,
            event_publisher=event_publisher,
            tool_registry=tool_registry,
            max_turns=agent_config.get("max_turns", 50),
        )

        await agent_loop.run(initial_message)

    finally:
        await event_publisher.disconnect()
        logger.info("Agent runtime shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
