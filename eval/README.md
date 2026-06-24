# eval/ — benchmark evaluation

One evaluation system, run by **[BenchFlow](https://github.com/benchflow-ai/benchflow)**
(the sole runner). A **benchmark** is a flat dir of **tasks** (each task = one
input → one trajectory → one score, à la
[skillsbench](https://github.com/benchflow-ai/skillsbench/tree/main/tasks)). The
**agent harness** and **model** are run-time knobs (`--agent` / `--model`), not
structure — so the same benchmark runs under pi, Claude Code, Codex, … against
any model.

```
eval/
  benchmarks/<name>/tasks/<id>/     a benchmark = a tasks/ dir; each task = task.md + verifier/
    simpleqa/  hotpotqa/            (browsecomp / gaia / hle come later)
  _tooling/                         run everything from here
    Makefile  scripts/  pi/  docs/
  _runs/<agent>/<benchmark>/        run outputs (gitignored)
  deep_research/                    PARKED — legacy dr_agent two-phase scaffold (see below)
```

## Run

```bash
cd eval/_tooling
make serve                                              # start the engine (SGLang :30000)
make gen BENCHMARK=simpleqa N=50                        # dataset -> benchmarks/simpleqa/tasks/
make eval BENCHMARK=simpleqa AGENT=pi MODEL=vllm/qwen35-2b-base
make eval BENCHMARK=hotpotqa AGENT=claude MODEL=claude-haiku-4-5-20251001
```

`AGENT` and `MODEL` are the only things that change to swap harness/model;
the benchmark (its `tasks/`) stays fixed. Outputs land in
`_runs/<agent>/<benchmark>/`. See [`_tooling/README.md`](_tooling/README.md) for
setup, the SGLang engine, and the known BenchFlow proxy issue
([`_tooling/docs/benchflow-engine-500.md`](_tooling/docs/benchflow-engine-500.md)).

## `deep_research/` is parked

[`deep_research/`](deep_research/CLAUDE.md) is the older dr_agent **two-phase**
(generate → score) harness — a separate runner. We're standardizing on BenchFlow
as the sole runner, so it's parked for now: when `libs/dr_agent` is wrapped as a
BenchFlow **agent**, "run dr_agent" becomes just another `--agent dr_agent` over
the same `benchmarks/`, and this folder folds in. Until then, ignore it.
