# rl_deepresearch

RL post-training recipe for the deep-research agent, built on
**[slime](https://github.com/THUDM/slime)** (Megatron + SGLang + Ray Buffer).

slime is the engine and is consumed as a Docker image — **we never fork it**. Our
entire contribution is two plugin functions wired in by CLI flag, mirroring
slime's own `examples/search-r1/`:

| Our file | slime flag | What it does |
|---|---|---|
| `plugins/rollout.py::generate` | `--custom-generate-function-path` | runs the shared `dr_agent` agent loop, fills the slime `Sample` contract (tokens, loss_mask, …) |
| `plugins/reward.py::reward_func` | `--custom-rm-path` | shims slime's `Sample` to the shared `dr_agent` reward registry |

Because both reuse `libs/dr_agent`, the RL environment and reward are **the same
code** that `eval/` and `apps/` run — no train/eval skew.

## Layout

```
plugins/      our rollout + reward (the only code we write)
configs/      versioned experiment YAMLs (base + experiments/)
launch/       run.sh — assembles Megatron/SGLang/slime arg blocks
engine/       README only — slime is a Docker image, not vendored
```

## Quickstart

See `engine/README.md` for the container workflow. In short:
`make image` → run the container with this recipe mounted → `bash launch/run.sh`.
