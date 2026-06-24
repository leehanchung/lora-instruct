# CLAUDE.md — eval/

Benchmark evaluation, run by **BenchFlow** (the sole runner). Work from
`eval/_tooling/` (its `Makefile` is the entrypoint); see [`README.md`](README.md)
for the model.

## Mental model (don't break it)

- **A benchmark = a flat `tasks/` dir** under `benchmarks/<name>/tasks/`. Each
  task dir (`task.md` + `verifier/`) is one input → one trajectory → one score.
- **BenchFlow is the runner, not a harness.** The harness is the **agent**
  (`pi` / `claude` / `codex` / …), selected at runtime via `--agent`; the model
  via `--model`. These are knobs, never folders — don't create per-agent or
  per-model directories or configs.
- **No per-benchmark `run.yaml`.** Run config (agent, model, sandbox, engine URL)
  is supplied by the `make eval` driver, so a benchmark folder is *just* tasks.

## Commands (from `eval/_tooling/`)

- `make eval BENCHMARK=<b> AGENT=<a> MODEL=<m>` — run a benchmark via BenchFlow
- `make gen BENCHMARK=<b> N=<n>` — generate task dirs from a dataset
- `make serve` — start the local engine (SGLang); `make smoke` — agent→engine check

## Conventions

- Adding a benchmark = add `benchmarks/<name>/tasks/` (a generator in
  `scripts/gen_tasks.py`), nothing else.
- `benchmarks/*/tasks/` and `_runs/` are gitignored (regenerable); keep generators
  and tooling, not bulk data, in git.
- One runner: **BenchFlow**. A different agent harness (dr_agent, Claude Code,
  Codex, …) is a `--agent` choice, never a parallel runner or a new folder.
