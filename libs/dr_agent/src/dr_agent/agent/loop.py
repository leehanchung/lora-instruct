"""The ReAct agent loop: generate -> tool call -> observe -> repeat.

This single loop is the one place the agent's behaviour is defined. It is reused
verbatim as:
  * the RL *environment* (training/rl_deepresearch/plugins/rollout.py wraps it),
  * the eval rollout generator (eval/harness/generate.py calls it),
  * the production runtime (apps/ call it).

Keeping it here is what guarantees train/eval/serve parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dr_agent.agent.config import AgentConfig


@dataclass
class Step:
    """One turn of the loop: the model's thought/action and the tool observation."""

    thought: str
    tool: str | None
    tool_args: dict | None
    observation: str | None


@dataclass
class AgentResult:
    """The full trajectory of one agent run.

    Carries enough to (a) extract a final answer for eval, and (b) reconstruct
    token-level masks for RL (so tool-observation tokens are excluded from the
    policy-gradient loss — see Search-R1 tensor_helper).
    """

    answer: str
    steps: list[Step] = field(default_factory=list)
    # Populated only in RL rollouts; the trainer plugin fills token ids + masks.
    raw: dict = field(default_factory=dict)


async def run_agent(task: str, config: AgentConfig) -> AgentResult:
    """Run the deep-research agent on a single task.

    TODO: implement the ReAct loop:
      1. render system prompt (dr_agent.prompts) + task
      2. call model provider (config.base_url / config.model)
      3. parse a tool call; if present, dispatch via dr_agent.tools over HTTP
         (config.tool_server_url) and append the observation
      4. repeat until <answer> or max_tool_calls / context_token_budget hit
    """
    raise NotImplementedError("agent loop not yet implemented")
