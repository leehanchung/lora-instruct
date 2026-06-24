"""Run driver: YAML run-config + benchmark task list -> generate -> score.

    uv run python -m eval_harness.harness.runner --config run.yaml --benchmark simpleqa

A run-config is one reproducible eval (which agent, which benchmark, output dir).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from dr_agent.agent import AgentConfig

from eval_harness.harness.generate import generate
from eval_harness.harness.score import score_rollouts


def _load_tasks(benchmark_dir: Path) -> list[dict]:
    # Each benchmark dir ships a tasks.jsonl loaded here (see benchmarks/<name>/).
    tasks_path = benchmark_dir / "tasks.jsonl"
    with tasks_path.open() as f:
        return [json.loads(line) for line in f]


async def _main_async(config_path: Path, benchmark: str) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    agent_config = AgentConfig(**cfg.get("agent", {}))

    # project root = eval/deep_research/ (…/src/eval_harness/harness/runner.py)
    project_root = Path(__file__).resolve().parents[3]
    benchmark_dir = project_root / "benchmarks" / benchmark
    out_dir = project_root / "results" / cfg.get("run_name", "run") / benchmark

    tasks = _load_tasks(benchmark_dir)
    rollouts = await generate(tasks, agent_config, out_dir)
    report = score_rollouts(rollouts)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{benchmark}: mean_score={report['mean_score']:.4f} (n={report['n']})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--benchmark", required=True)
    args = p.parse_args()
    asyncio.run(_main_async(args.config, args.benchmark))


if __name__ == "__main__":
    main()
