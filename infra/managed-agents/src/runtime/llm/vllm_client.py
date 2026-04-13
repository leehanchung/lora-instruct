"""vLLM / OpenAI-compatible API client for agent runtime."""

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from .base import LLMClient, LLMResponse, TextBlock, ToolUseBlock, Usage

logger = logging.getLogger(__name__)


class VLLMClient(LLMClient):
    """vLLM / OpenAI-compatible API client."""

    def __init__(self, config: dict[str, Any]):
        """Initialize vLLM client.

        Args:
            config: Config dict with:
                - model: Model name
                - base_url: vLLM endpoint URL (e.g., http://localhost:8000/v1)
                - api_key: Optional API key (defaults to "not-used" for local vLLM)
                - Other optional params
        """
        self.model = config.get("model", "meta-llama/Llama-2-7b-chat-hf")
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        api_key = config.get("api_key", "not-used")

        self.client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        logger.info(f"Initialized vLLM client with model: {self.model} at {self.base_url}")

    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for OpenAI-compatible API.

        Args:
            tools: List of tool definitions

        Returns:
            Formatted function definitions for API
        """
        functions = []
        for tool in tools:
            functions.append(
                {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                }
            )
        return functions

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Get completion from vLLM.

        Args:
            messages: Message history
            tools: Available tools
            system: System prompt
            max_tokens: Max response tokens

        Returns:
            LLMResponse
        """
        # Add system message if provided
        messages_to_send = []
        if system:
            messages_to_send.append({"role": "system", "content": system})
        messages_to_send.extend(messages)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages_to_send,
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": func} for func in self._format_tools(tools)
            ]
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)

        # Parse response content
        content = []
        choice = response.choices[0]

        # Handle text content
        if choice.message.content:
            content.append(TextBlock(type="text", text=choice.message.content))

        # Handle tool calls
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                # Parse tool input JSON
                tool_input = {}
                if isinstance(tool_call.function.arguments, str):
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse tool arguments: {tool_call.function.arguments}")
                        tool_input = {"_raw": tool_call.function.arguments}
                else:
                    tool_input = tool_call.function.arguments or {}

                content.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=tool_input,
                    )
                )

        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=content,
            stop_reason=choice.finish_reason or "end_turn",
            usage=usage,
        )
