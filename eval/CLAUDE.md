# CLAUDE.md — dr-eval (evaluation harness)

Standalone two-phase eval (generate → score). Reuses `libs/dr_agent` for both the
agent loop and the scorers, so eval can never drift from training.

## Commands

- `make eval BENCHMARK=<name> CONFIG=<run.yaml>` — generate + score a benchmark
- `make check` / `make test`

## Conventions

- **Keep generate and score separate.** Never fold scoring into generation —
  offline re-scoring is the whole point.
- **One folder per benchmark** under `benchmarks/`, identical layout
  (`config.yaml`, `tasks.jsonl`, optional `grader.py`).
- **Scorers live in `libs/dr_agent/rewards`**, not here. Add a `grader.py` only
  for a genuinely benchmark-specific metric, and prefer promoting it to the
  shared registry.
- **Shared sampling logic goes in `samplers/`**, never copy-pasted per benchmark.
- `results/` is gitignored; never commit rollout dumps.
