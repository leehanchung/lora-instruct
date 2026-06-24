# simpleqa benchmark

One folder per benchmark (DR-Tulu / QUEST pattern). Each folder holds:

- `config.yaml` — benchmark name + which `data_source` scorer to use
- `tasks.jsonl` — the tasks: `{"id", "prompt", "ground_truth", "data_source"}`
  per line (add this file; keep large datasets out of git via `.gitignore`)
- optional `grader.py` — only if the benchmark needs a bespoke grader beyond the
  shared `dr_agent.rewards` registry

Run it:

```bash
uv run python -m eval_harness.harness.runner --config run.yaml --benchmark simpleqa
```

`browsecomp/`, `gaia/`, `hle/` follow the identical layout.
