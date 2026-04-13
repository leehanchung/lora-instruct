"""Anthropic API client for agent runtime."""

import json
import logging
from typing import Any, Optional

from anthropic import AsyncAnthropic

from .base import LLMClient, LLMResponse, TextBlock, ToolUseBlock, Usage

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Anthropic client.

        Args:
            config: Config dict with:
                - api_key: Anthropic API key
                - model: Model name (e.g., "claude-3-sonnet-20240229")
                - Other optional params
        """
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-3-5-sonnet-20241022")
        self.client = AsyncAnthropic(api_key=self.api_key)
        logger.info(f"Initialized Anthropic client with model: {self.model}")

    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for Anthropic API.

        Args:
            tools: List of tool definitions

        Returns:
            Formatted tool list for API
        """
        formatted = []
        for tool in tools:
            formatted.append(
                {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema", {"type": "object", "properties": {}}),
                }
            )
        return formatted

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Get completion from Claude.

        Args:
            messages: Message history
            tools: Available tools
            system: System prompt
            max_tokens: Max response tokens

        Returns:
            LLMResponse
        """
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = self._format_tools(tools)

        response = await self.client.messages.create(**kwargs)

        # Parse response content
        content = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(type="text", text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return LLMResponse(
            content=content,
            stop_reason=response.stop_reason,
            usage=usage,
        )
