"""Abstract base classes and factories for LLM clients."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Text content block."""

    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    """Tool use block."""

    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class Usage:
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: list[TextBlock | ToolUseBlock]
    stop_reason: str = "end_turn"
    usage: Optional[Usage] = None


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Get completion from LLM.

        Args:
            messages: Message history in format [{"role": "user"|"assistant", "content": "..."}]
            tools: List of available tools with schema
            system: System prompt
            max_tokens: Max tokens in response

        Returns:
            LLMResponse with content, stop_reason, and usage
        """
        pass


def create_llm_client(config: dict[str, Any]) -> LLMClient:
    """Factory function to create appropriate LLM client.

    Args:
        config: Configuration dict with keys:
            - provider: "anthropic" or "vllm"
            - model: model name
            - api_key: API key (for Anthropic)
            - base_url: base URL (for vLLM)
            - Other provider-specific config

    Returns:
        LLMClient instance

    Raises:
        ValueError: If provider not supported
    """
    provider = config.get("provider", "anthropic").lower()

    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(config)
    elif provider == "vllm" or provider == "openai":
        from .vllm_client import VLLMClient

        return VLLMClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
