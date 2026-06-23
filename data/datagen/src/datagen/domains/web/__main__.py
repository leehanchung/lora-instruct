"""Orchestrate the web data-gen pipeline: explore -> verify -> distract -> extend.

Each stage is independently runnable; this wires them in order. Stages write to
SEPARATE output dirs (raw -> verified -> final) for clean partial reruns and
provenance, rather than mutating one dir in place.

    uv run python -m datagen.domains.web --seeds seeds.txt --out outputs/web
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from datagen.core import Task
from datagen.domains.web.distract import WebDistractor
from datagen.domains.web.explore import WebExplorer
from datagen.domains.web.extend import WebExtender
from datagen.domains.web.verify import WebVerifier


def _write(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")


async def run_pipeline(seeds: list[str], out: Path) -> None:
    tasks = [Task(id=f"web-{i}", question="", truth="", level=0) for i, _ in enumerate(seeds)]
    tasks = await WebExplorer().run(tasks)
    _write(tasks, out / "raw" / "tasks.jsonl")
    tasks = await WebVerifier().run(tasks)
    _write(tasks, out / "verified" / "tasks.jsonl")
    tasks = await WebDistractor().run(tasks)
    tasks = await WebExtender().run(tasks)
    _write(tasks, out / "final" / "tasks.jsonl")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    seeds = [s for s in args.seeds.read_text().splitlines() if s.strip()]
    asyncio.run(run_pipeline(seeds, args.out))


if __name__ == "__main__":
    main()
