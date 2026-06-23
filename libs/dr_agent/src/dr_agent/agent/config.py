"""Agent configuration — the thin binding of prompt + tools + model provider.

Kept as a dataclass-style pydantic model so the *same* config object can be built
by a serving app, by the eval harness, and by the RL rollout plugin.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelParams(BaseModel):
    """Sampling / inference parameters passed to the model provider."""

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096


class AgentConfig(BaseModel):
    """Everything needed to instantiate a concrete deep-research agent.

    The framework (this lib) knows nothing about "deep research" specifically;
    a concrete agent is this config: a system prompt, a tool selection, a model
    provider endpoint, and sampling params. See StepFun cortex-vs-demo split.
    """

    # Model provider — an OpenAI-compatible endpoint (vLLM/SGLang/hosted API).
    model: str = "claude-opus-4-8"
    base_url: str | None = None
    params: ModelParams = Field(default_factory=ModelParams)

    # Which tools this agent may call (names resolved against dr_agent.tools).
    tools: list[str] = Field(default_factory=lambda: ["search", "visit"])

    # Endpoint of the search/tool HTTP service (services/search_server).
    tool_server_url: str = "http://127.0.0.1:8000"

    # Prompt template name under dr_agent/prompts/.
    system_prompt: str = "deep_research.jinja"

    # Long-horizon controls.
    max_tool_calls: int = 20
    context_token_budget: int = 80_000
