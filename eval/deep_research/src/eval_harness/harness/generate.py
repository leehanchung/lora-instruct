"""Phase 1: generate rollouts.

Runs the SAME dr_agent loop the RL trainer uses, over a benchmark's task list,
and writes one rollout JSON per task to results/<run>/rollouts/. No scoring here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dr_agent.agent import AgentConfig, run_agent


async def _run_one(task: dict, config: AgentConfig) -> dict:
    result = await run_agent(task["prompt"], config)
    return {
        "id": task["id"],
        "prompt": task["prompt"],
        "data_source": task.get("data_source", "f1"),
        "ground_truth": task.get("ground_truth", ""),
        "prediction": result.answer,
        "steps": [s.__dict__ for s in result.steps],
    }


async def generate(tasks: list[dict], config: AgentConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts = await asyncio.gather(*(_run_one(t, config) for t in tasks))
    out = out_dir / "rollouts.jsonl"
    with out.open("w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")
    return out
