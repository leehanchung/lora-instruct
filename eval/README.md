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

## Other harnesses?

There's one runner: **BenchFlow**. To run a different agent harness
(`dr_agent`, Claude Code, Codex, …) on these benchmarks, register it as a
BenchFlow **agent** and select it with `--agent` — the benchmarks (`tasks/`) and
the runner don't change. The shared `libs/dr_agent` loop is the intended path for
a future `--agent dr_agent`; until that wrapper exists, the dr_agent harness isn't
wired here.
