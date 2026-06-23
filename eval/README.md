# dr-eval

Standalone **two-phase** evaluation harness for deep-research agents:

1. **generate** — run the shared `dr_agent` loop over a benchmark's tasks, write
   rollout JSONL (`harness/generate.py`);
2. **score** — score those rollouts offline with the shared `dr_agent` reward
   registry (`harness/score.py`).

Decoupling generation from scoring means you can re-score (new metric, fixed
grader) without paying to re-generate. It also means eval uses **the exact same
agent loop and scorers as RL training** — no train/eval skew.

> This is deliberately a standalone top-level category, not buried inside a
> training recipe. Search-R1 / DeepResearcher collapsed eval into "val batches +
> reward inside the trainer," which makes eval impossible to reproduce outside a
> training run. Don't repeat that.

## Layout

```
src/eval_harness/
  harness/   generate.py · score.py · runner.py
  samplers/  shared sampling logic (one place, not per-benchmark)
benchmarks/  one folder per benchmark (browsecomp · gaia · hle · simpleqa)
results/     gitignored rollout + report artifacts
```

## Run

```bash
make eval BENCHMARK=simpleqa CONFIG=run.yaml
```
