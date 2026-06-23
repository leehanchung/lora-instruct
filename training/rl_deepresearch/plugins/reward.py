"""Custom slime reward: a thin shim to the shared dr_agent reward registry.

Wired into slime via `--custom-rm-path plugins.reward.reward_func`.

The actual scoring logic lives in libs/dr_agent/rewards (the SAME registry the
eval harness uses), so a reward improvement is picked up by training and eval at
once. This file only adapts slime's `Sample` to a `dr_agent` `Row`.
"""

from __future__ import annotations

from dr_agent.rewards import Row, score


async def reward_func(args, sample, **kwargs) -> float:
    """slime custom reward.

    Signature required by slime:
        async def reward_func(args, sample: Sample, **kwargs) -> float

    `sample.label` carries per-example ground truth; `sample.label["data_source"]`
    selects the scorer in the registry.
    """
    label = sample.label or {}
    row = Row(
        data_source=label.get("data_source", "f1"),
        prediction=sample.response,
        ground_truth=label.get("ground_truth", ""),
        extra=label.get("extra", {}),
    )
    return score(row).score
