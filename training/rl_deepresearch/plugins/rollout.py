"""Custom slime rollout: drive the shared dr_agent loop as the RL environment.

Wired into slime via `--custom-generate-function-path plugins.rollout.generate`.
This is the recommended slime integration point (override the per-sample
generation step, NOT the whole rollout loop) — see slime examples/search-r1/.

The agent loop itself lives in libs/dr_agent; this file's only job is to:
  1. run that loop on the sample's prompt,
  2. tokenize prompt + response,
  3. build a loss_mask that ZEROES tool/observation tokens (so only model-
     generated tokens get a gradient — see Search-R1 tensor_helper), and
  4. populate the slime `Sample` contract.
"""

from __future__ import annotations

# NOTE: `Sample` and the SGLang sampling params are provided by the slime runtime
# inside the container; imported lazily so this package is importable for linting
# outside the image.
from dr_agent.agent import AgentConfig, run_agent


def _agent_config_from_args(args) -> AgentConfig:
    """Build the agent config from slime's args Namespace."""
    return AgentConfig(
        model=getattr(args, "hf_checkpoint", "policy"),
        base_url=getattr(args, "sglang_router_url", None),
        tool_server_url=getattr(args, "tool_server_url", "http://127.0.0.1:8000"),
        max_tool_calls=getattr(args, "max_tool_calls", 20),
    )


async def generate(args, sample, sampling_params):
    """slime per-sample generate function.

    Signature required by slime:
        async def generate(args, sample: Sample, sampling_params) -> Sample
    """
    config = _agent_config_from_args(args)

    # 1. Run the shared deep-research agent loop on the prompt.
    result = await run_agent(sample.prompt, config)  # noqa: F841  (used once implemented)

    # 2-4. TODO: tokenize prompt+response, build loss_mask (mask tool-observation
    # tokens), and fill the Sample contract:
    #   sample.tokens            = prompt_token_ids + response_token_ids
    #   sample.response          = result.answer / full decoded response
    #   sample.response_length   = len(response_token_ids)
    #   sample.loss_mask         = [0 for prompt/tool tokens, 1 for model tokens]
    #   sample.status            = Sample.Status.COMPLETED | TRUNCATED | ABORTED
    #   sample.rollout_log_probs = ... (optional)
    raise NotImplementedError("fill the slime Sample contract from AgentResult")
