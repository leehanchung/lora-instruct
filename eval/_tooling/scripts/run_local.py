#!/usr/bin/env python3
"""Docker-free local fallback runner.

BenchFlow requires a Docker/Daytona/Modal sandbox. When none is available, this
runner reproduces the *scoring* half of the eval without a sandbox: it reads the
generated BenchFlow task dirs, asks Qwen the question **through the pi harness**
(`pi -p --no-tools`), and scores each answer with that task's own
`verifier/score.py`. Same tasks, same gold, same scorer as the BenchFlow path —
only the sandbox/agentic-tool layer is skipped.

Use `bench eval run` (the real harness) once Docker is up; use this for a quick,
Docker-free signal.

    python scripts/run_local.py --benchmark simpleqa --n 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent       # eval/_tooling/scripts
EVAL_ROOT = HERE.parents[1]                   # eval/
PI = os.environ.get("PI", str(Path.home() / ".npm-global/bin/pi"))
NODE_BIN = str(Path.home() / ".nvm/versions/node/v22.23.0/bin")
MODEL = os.environ.get("PI_MODEL", "sglang/qwen35-2b-base")


def question_from_task(task_md: Path) -> str:
    text = task_md.read_text()
    m = re.search(r"Question:\s*(.+)", text)
    return m.group(1).strip() if m else ""


def ask_pi(question: str, timeout: int) -> str:
    prompt = (
        f"{question}\n\nAnswer with ONLY a short factual phrase — no explanation."
    )
    env = {**os.environ, "PATH": f"{NODE_BIN}:{os.environ.get('PATH', '')}"}
    try:
        out = subprocess.run(
            [PI, "-p", "--model", MODEL, "--no-tools", prompt],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return out.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def score_one(task_dir: Path, answer: str) -> float:
    verifier = task_dir / "verifier"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(answer)
        ans_path = f.name
    out = subprocess.run(
        ["python3", str(verifier / "score.py"), str(verifier / "gold.json"),
         ans_path, "/nonexistent.jsonl"],
        capture_output=True, text=True,
    )
    os.unlink(ans_path)
    try:
        return float(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["simpleqa", "hotpotqa"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    bench_root = EVAL_ROOT / "benchmarks" / args.benchmark / "tasks"
    task_dirs = sorted(p for p in bench_root.iterdir() if p.is_dir())[: args.n]

    rows, total = [], 0.0
    for task_dir in task_dirs:
        question = question_from_task(task_dir / "task.md")
        answer = ask_pi(question, args.timeout)
        gold = json.loads((task_dir / "verifier" / "gold.json").read_text())["answer"]
        s = score_one(task_dir, answer)
        total += s
        rows.append({"id": task_dir.name, "score": s, "answer": answer, "gold": gold})
        print(f"  {task_dir.name}: {s:.2f}  pred={answer[:50]!r}  gold={gold[:40]!r}")

    n = len(rows)
    mean = total / n if n else 0.0
    out_dir = EVAL_ROOT / "_runs" / "_direct-debug" / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps({"benchmark": args.benchmark, "n": n, "mean_score": mean,
                    "rows": rows}, indent=2)
    )
    print(f"\n{args.benchmark}: mean_score={mean:.4f} (n={n})  -> {out_dir}/report.json")


if __name__ == "__main__":
    main()
