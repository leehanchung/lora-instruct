# Plan: Deep Agent Training & Evaluation with Reinforcement Learning

> Transforming LoRA-Instruct into a deep agent RL training and evaluation repository, with autoresearch-style autonomous experimentation.

## 1. Vision and Goals

The goal is to evolve this repository from a LoRA fine-tuning toolkit into a full-stack platform for training LLM agents via reinforcement learning (RL), then layering on Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern so an AI agent can autonomously experiment with RL training recipes overnight.

**Phased approach:**

1. **Phase 1 (Now): Tinker API + Evaluation.** Use [Tinker](https://tinker-docs.thinkingmachines.ai/) (Thinking Machines' managed RL training API) to get GRPO training running fast — no local training loop to build. Focus effort on evaluation environments, reward functions, and the autoresearch experiment loop.
2. **Phase 2 (Future): Migrate to prime-rl.** Adopt [PrimeIntellect's prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for self-hosted async RL training at scale, keeping the evaluation and autoresearch infrastructure built in Phase 1.

**Why this ordering:** Tinker handles all distributed training infrastructure (LoRA, GRPO, rollouts, optimization) via API calls, letting us validate the evaluation + autoresearch loop without writing a training loop. Once we've proven the outer loop works, we swap the training backend to prime-rl for full control and scale.

---

## 2. Current State Assessment

### What we have (LoRA-Instruct)

| Component | Status | Reusable? |
|---|---|---|
| LoRA fine-tuning (`finetune.py`) | Working | Yes — reference for SFT, eventually replaced |
| Prompt system (`utils/prompter.py`) | Working | Yes — extend for chat/agent templates |
| Inference benchmarking (`inference/bench.py`) | Working | Partially — latency metrics, not agent eval |
| Alpaca datasets (`dataset/`) | Working | Yes — seed SFT data |
| DDP support | Working | Archive — Tinker/prime-rl handle distribution |
| Notebooks | Working | Archive or adapt |
| RL / reward modeling / agent evaluation | **Missing** | N/A |

### External systems we'll integrate

| System | Role | Phase |
|---|---|---|
| **Tinker API** | Managed LoRA training + GRPO + sampling. Handles compute, no local GPU needed for training. | Phase 1 |
| **prime-rl** | Self-hosted async RL framework (FSDP2 + vLLM). Orchestrator → Trainer → Inference architecture. | Phase 2 |
| **autoresearch** | Design pattern: autonomous agent experiment loop with git-based tracking. | Both phases |

---

## 3. Phase 1: Tinker API + Evaluation + Autoresearch

### 3.1 How Tinker Works

Tinker is a remote training API from Thinking Machines (Mira Murati's lab). You write Python that calls their API — they handle the distributed GPU infrastructure. Key concepts:

```python
import tinker

svc = tinker.ServiceClient()  # Authenticates via TINKER_API_KEY

# Create a LoRA training client on a remote GPU
tc = svc.create_lora_training_client(base_model="Qwen/Qwen3-8B", rank=32)

# GRPO: generate completions, compute rewards, train
sc = tc.save_weights_and_get_sampling_client()
response = sc.sample(prompt=model_input, num_samples=group_size,
                     sampling_params=SamplingParams(max_tokens=256))

# Compute rewards locally, then send training data back
tc.forward_backward(data=training_data, loss_fn="importance_sampling")
tc.optim_step(adam_params=AdamParams(learning_rate=4e-5))
```

**What Tinker gives us for free:**
- LoRA fine-tuning on any supported open-source model (Llama, Qwen, etc.)
- GRPO training with `importance_sampling` loss (also PPO, CISPO, DRO)
- Async training — overlap rollouts with GPU training for throughput
- Sampling/generation from the current policy
- Checkpoint save/load/download
- No local GPU needed for training

**What we still need to build:**
- Evaluation environments and reward functions
- The autoresearch experiment loop
- Custom `ProblemEnv` / `MessageEnv` implementations for our tasks
- Data pipeline (prompt datasets, eval sets)

### 3.2 Tinker's Environment System

Tinker provides a clean abstraction for RL environments:

**`ProblemEnv` — single-turn answer verification (math, QA):**
```python
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder

class MathEnv(ProblemEnv):
    def get_question(self) -> str: ...
    def check_answer(self, sample_str: str) -> bool: ...
    def check_format(self, sample_str: str) -> bool: ...
    def get_reference_answer(self) -> str: ...
```
Reward formula: `format_coef * (check_format - 1) + check_answer`

**`MessageEnv` — multi-turn conversations (tool use, games):**
```python
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult

class ToolUseEnv(MessageEnv):
    async def initial_observation(self) -> list[Message]: ...
    async def step(self, message: Message) -> MessageStepResult: ...
```

**Built-in environments we can use immediately:**
- `Gsm8kDatasetBuilder` — grade-school math
- `ArithmeticDatasetBuilder` — simple arithmetic
- `DeepcoderDatasetBuilder` — code generation with sandbox

### 3.3 Evaluation Framework

Build `eval/` with pluggable evaluation environments. These are independent of the training backend — they work with Tinker now and prime-rl later.

| Environment | Type | Metric | Description |
|---|---|---|---|
| **Math reasoning** (GSM8K, MATH) | ProblemEnv | Accuracy | Parse final answer, compare to ground truth |
| **Code generation** (HumanEval, MBPP) | ProblemEnv | pass@k | Execute generated code against test cases |
| **Tool use** (calculator, search) | MessageEnv | Task completion rate | Multi-turn: agent calls tools, gets results |
| **Instruction following** (IFEval) | ProblemEnv | Constraint satisfaction | Does output follow formatting/content rules? |

**Primary metric for autoresearch:** Single scalar `eval_score` — start with GSM8K accuracy alone, later move to a weighted composite.

### 3.4 Autoresearch Adaptation for Tinker

The autoresearch pattern adapts naturally to Tinker. The key difference: instead of modifying a local `train.py` that runs on a GPU, the agent modifies a script that makes Tinker API calls.

**Three-file architecture:**
```
prepare.py          # Fixed: data prep, eval harness, Tinker client setup
                    # Downloads datasets, defines compute_score()
                    # NOT modified by the agent

rl/train.py         # Agent-editable: Tinker training recipe
                    # Everything is fair game: model choice, GRPO hyperparams,
                    # group_size, learning rate, reward shaping, max_tokens, etc.
                    # Calls Tinker API for training + sampling

program.md          # Human-editable: agent instructions
```

**The experiment loop:**
```
SETUP:
1. Create Tinker training client with base model + LoRA
2. Run eval to establish baseline score
3. Initialize results.tsv

LOOP:
1. Agent reads current state: git branch, results.tsv, recent experiments
2. Agent modifies rl/train.py with an experimental idea
   (e.g., "try group_size=16 instead of 8", "switch to Qwen3-8B",
    "add format reward for chain-of-thought", "increase max_tokens to 512")
3. git commit
4. Run: python rl/train.py > run.log 2>&1
5. Read results: grep "^eval_score:" run.log
6. If eval_score improved → keep (advance branch)
7. If eval_score same or worse → git reset
8. Log to results.tsv
9. GOTO 1
```

**Experiment budget:** Tinker API calls have latency but no fixed GPU time budget. We enforce a wall-clock budget (e.g., 10-15 min per experiment) that covers N training steps + eval. This is tunable in `prepare.py`.

### 3.5 What the Agent Can Experiment With

| Category | Examples |
|---|---|
| Model selection | `Llama-3.2-1B`, `Qwen3-8B`, `Qwen3-30B-A3B` (MoE, cost-efficient) |
| LoRA config | rank (16, 32, 64), target modules (attn, mlp, unembed) |
| GRPO hyperparams | group_size (4, 8, 16), learning_rate (1e-5 to 4e-5), kl_penalty_coef |
| Loss function | `importance_sampling`, `ppo`, `cispo`, `dro` |
| Generation | temperature, max_tokens, top_p |
| Reward design | Format reward coefficient, reward composition |
| Training dynamics | Steps per experiment, batch size, LR schedule |
| Async config | `max_steps_off_policy`, `groups_per_batch` |

---

## 4. Phase 2: Migration to prime-rl (Future)

### 4.1 Why prime-rl

Once the evaluation and autoresearch infrastructure is proven with Tinker, we migrate to prime-rl for:

- **Full control over the training loop** — modify optimizers, loss functions, model architectures directly.
- **Scale to 1000+ GPUs** with FSDP2 + vLLM async architecture.
- **Self-hosted** — no API costs, no dependency on external service availability.
- **Agentic multi-turn training** — prime-rl is designed for async agentic RL with `verifiers` environments.
- **Custom algorithms** — bring your own loss functions, not limited to Tinker's supported set.

### 4.2 prime-rl Architecture (for reference)

prime-rl has three components that run as separate processes:

```
┌─────────────┐    rollouts    ┌──────────────┐   packed batches   ┌─────────────┐
│  Inference   │ ──────────→   │ Orchestrator  │ ──────────────→    │   Trainer   │
│  (vLLM)     │ ←──────────   │  (CPU, async) │ ←──────────────    │  (FSDP2)   │
│             │   new weights  │              │   updated weights   │            │
└─────────────┘               └──────────────┘                     └─────────────┘
```

- **Orchestrator:** Collects rollouts, computes advantages, dispatches batches, relays weights.
- **Trainer:** FSDP2-distributed training, receives packed batches, updates model.
- **Inference:** vLLM-based generation server, serves current (or slightly stale) policy.

**Key prime-rl features we'll use:**
- LoRA support (including multi-LoRA for parallel experiments)
- `verifiers` environment integration
- TOML config system
- Async off-policy training (up to k steps of staleness)
- Built-in examples: reverse text, wordle, alphabet sort, wiki search, math

### 4.3 Migration Path

The migration is clean because we're separating concerns:

| Component | Phase 1 (Tinker) | Phase 2 (prime-rl) |
|---|---|---|
| Training backend | Tinker API calls | prime-rl trainer + orchestrator + inference |
| RL algorithm | Tinker's `importance_sampling` / `ppo` | prime-rl's loss functions (IPO, custom) |
| Generation | Tinker `SamplingClient` | prime-rl vLLM inference server |
| **Eval environments** | **Our code (portable)** | **Same code, unchanged** |
| **Reward functions** | **Our code (portable)** | **Same code, unchanged** |
| **Autoresearch loop** | **Our code (portable)** | **Same code, minor edits to train.py** |
| Config | Python kwargs to Tinker | TOML files for prime-rl |

The eval/, reward, and autoresearch infrastructure transfers directly. Only `rl/train.py` changes — from Tinker API calls to prime-rl config + launch commands.

### 4.4 prime-rl Environment Compatibility

prime-rl uses `verifiers` environments from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments). Our Tinker `ProblemEnv` / `MessageEnv` implementations can be adapted to `verifiers` format. Plan:

1. Build eval environments using Tinker's `ProblemEnv` interface first (simpler, faster iteration).
2. When migrating, wrap them as `verifiers` environments for prime-rl compatibility.
3. Alternatively, build directly on `verifiers` if we want to skip the Tinker-specific layer.

---

## 5. Proposed Directory Structure

```
lora-instruct/
├── prepare.py                  # NEW: fixed data prep + eval harness
├── program.md                  # NEW: autoresearch agent instructions
├── results.tsv                 # NEW: experiment log (untracked by git)
│
├── rl/
│   ├── train.py                # THE agent-editable file for autoresearch
│   │                           # Phase 1: Tinker API calls
│   │                           # Phase 2: prime-rl config + launch
│   ├── envs/
│   │   ├── math_env.py         # GSM8K / MATH environment (ProblemEnv)
│   │   ├── code_env.py         # Code generation environment
│   │   ├── tool_env.py         # Tool use environment (MessageEnv)
│   │   └── instruction_env.py  # Instruction following environment
│   └── rewards.py              # Reward function interfaces + composites
│
├── eval/
│   ├── harness.py              # Unified evaluation runner
│   ├── datasets.py             # Eval dataset loading (GSM8K, HumanEval, etc.)
│   └── metrics.py              # Metric computation + composite scoring
│
├── sft/
│   ├── train.py                # Refactored from finetune.py (kept for reference)
│   └── configs/                # SFT config presets
│
├── utils/
│   ├── prompter.py             # Extended prompt templates (existing)
│   ├── callbacks.py            # Streaming helpers (existing)
│   └── tinker_utils.py         # NEW: Tinker API helpers and wrappers
│
├── templates/
│   ├── alpaca.json             # Existing
│   ├── chatml.json             # NEW: ChatML format
│   └── reasoning.json          # NEW: with <think> tags for reasoning
│
├── finetune.py                 # Existing (kept, still works)
├── inference/                  # Existing benchmarking (keep as-is)
├── dataset/                    # Existing SFT datasets
├── notebook/                   # Existing notebooks
├── docs/
│   ├── architecture.md         # Updated
│   ├── development.md          # Updated
│   └── plan-deep-agent-rl.md   # This document
│
├── pyproject.toml              # Updated dependencies
└── Research/                   # Autoresearch experiment branches and analysis
```

---

## 6. Dependencies

### Phase 1 (Tinker)

```toml
# Tinker SDK + cookbook
tinker = ">=0.1.0"
tinker-cookbook = {git = "https://github.com/thinking-machines-lab/tinker-cookbook.git"}

# Evaluation
datasets = ">=3.0.0"           # HuggingFace datasets (GSM8K, etc.)

# Existing (keep)
transformers = ">=4.45.0"
peft = ">=0.14.0"
accelerate = ">=1.0.0"
```

### Phase 2 (prime-rl, future)

```toml
# prime-rl framework
prime-rl = {git = "https://github.com/PrimeIntellect-ai/prime-rl.git"}
# Requires: torch, vllm, flash-attn, verifiers
```

---

## 7. Implementation Roadmap

### Milestone 1: Tinker + GSM8K Baseline (Week 1-2)

**Goal:** GRPO training on GSM8K via Tinker, with eval and autoresearch scaffolding.

1. Set up Tinker API key and verify connectivity.
2. Build `rl/envs/math_env.py` — GSM8K `ProblemEnv` using Tinker's interface.
3. Build `eval/harness.py` — evaluate a model on GSM8K via Tinker `SamplingClient`.
4. Build `rl/train.py` — minimal GRPO training script using Tinker's `train.main()` or manual loop.
5. Run baseline: establish GSM8K accuracy for an untrained model, then after N GRPO steps.

### Milestone 2: Autoresearch Loop (Week 3-4)

**Goal:** Autonomous agent can run experiments overnight on Tinker.

6. Write `prepare.py` — downloads GSM8K data, defines `compute_score()`.
7. Write `program.md` — full agent instructions for Tinker-based RL experiments.
8. Ensure `rl/train.py` emits `eval_score:` line and respects time budget.
9. Set up git branching workflow (`autoresearch/<tag>`).
10. Test: run 5-10 autonomous experiments, verify keep/discard logic.

### Milestone 3: Expanded Environments (Week 5-6)

**Goal:** Multiple eval environments, composite scoring.

11. Build `rl/envs/tool_env.py` — multi-turn calculator/search tool use (`MessageEnv`).
12. Build `rl/envs/code_env.py` — code generation with Tinker's `DeepcoderDatasetBuilder` or custom.
13. Implement composite scoring in `eval/metrics.py`.
14. Update `program.md` with guidance for multi-environment optimization.
15. Stress-test: overnight autoresearch run (target: 30+ experiments).

### Milestone 4: prime-rl Migration (Future)

**Goal:** Self-hosted RL training with full control.

16. Set up prime-rl on GPU cluster (or single GPU for dev).
17. Port `ProblemEnv`/`MessageEnv` implementations to `verifiers` environments.
18. Rewrite `rl/train.py` to launch prime-rl components (orchestrator + trainer + inference).
19. Write TOML configs for our training setups.
20. Verify: same eval improvements as Tinker, but self-hosted.
21. Scale: multi-GPU async training, larger models, longer experiments.

---

## 8. Key Design Decisions

### 8.1 Tinker First, prime-rl Later

**Rationale:** Tinker lets us skip the hardest part (building a distributed RL training loop) and focus on what matters most: evaluation environments, reward design, and the autoresearch loop. Once those are validated, swapping the training backend is a targeted change to one file (`rl/train.py`).

**Trade-off:** We depend on Tinker's API for Phase 1 (cost, availability, supported models). But Phase 1 is about proving the concept, not production scale.

### 8.2 GRPO as Default Algorithm

**Rationale:** Both Tinker and prime-rl support GRPO natively. It eliminates the critic model (memory-efficient), works well with verifiable rewards (math, code), and is the standard since DeepSeek-R1. Tinker uses `importance_sampling` loss for GRPO; prime-rl uses its IPO variant.

### 8.3 LoRA Throughout

**Rationale:** Tinker only does LoRA fine-tuning (by design — they believe LoRA matches full fine-tuning for RL). prime-rl also supports LoRA natively. This is our project's core identity and makes everything faster and cheaper.

### 8.4 Evaluation as the Stable Core

**Rationale:** The eval framework (`eval/`, `rl/envs/`, `rl/rewards.py`) is the one piece that doesn't change between Tinker and prime-rl. Building it well in Phase 1 means Phase 2 migration is just a training backend swap. This is also what autoresearch optimizes against — it's the ground truth.

### 8.5 MoE Models for Cost Efficiency

**Rationale:** Tinker bills by active parameters. MoE models like `Qwen3-30B-A3B` (3B active out of 30B total) give near-large-model quality at small-model cost. Prefer these for Phase 1 experiments.

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Tinker API costs add up during autoresearch | Budget overrun from 50+ experiments | Start with small models (1-3B active); set per-experiment step limits; monitor costs |
| Tinker API availability/rate limits | Experiments stall overnight | Add retry logic; autoresearch handles failures gracefully (log crash, move on) |
| Evaluation too slow via Tinker sampling | Time budget blown on eval, not training | Use small eval subsets (200 problems); cache baseline completions |
| Tinker → prime-rl migration harder than expected | Environments don't port cleanly | Design environments against a thin interface; abstract Tinker-specific bits |
| Reward hacking | Agent games the metric | Use verifiable rewards; composite scoring; human review of autoresearch logs |

---

## 10. Success Criteria

### Phase 1 (Tinker)

1. **GRPO training runs end-to-end** on GSM8K via Tinker API.
2. **RL improves over baseline** on GSM8K (target: +5% accuracy after N steps).
3. **Autoresearch runs 30+ experiments** overnight without human intervention.
4. **At least 15% of experiments are "keep"** (meaningful improvement rate).
5. **At least 2 eval environments** working (math + one of: code, tool use, instruction).

### Phase 2 (prime-rl, future)

6. **Same eval improvements** reproduced on self-hosted prime-rl.
7. **Scale to 4+ GPUs** with async training.
8. **Autoresearch runs 50+ experiments** overnight.
9. **Multi-turn agent training** working (tool use or similar).

---

## 11. References

### Core Systems

- [Tinker API](https://tinker-docs.thinkingmachines.ai/) — managed RL training API (Phase 1 backend)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) — recipes, environments, and skills for Tinker
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — async RL training at scale (Phase 2 backend)
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous experiment loop pattern

### RL Algorithms & Research

- [DeepSeek-R1](https://arxiv.org/abs/2401.12954) — GRPO for reasoning model training
- [Tree-GRPO (ICLR 2026)](https://github.com/AMAP-ML/Tree-GRPO) — tree-search rollout strategy for agent RL
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — reference for scalable RL architecture
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — HuggingFace RL library
- [ART (Agent Reinforcement Trainer)](https://github.com/OpenPipe/ART) — GRPO for multi-step agents

### Evaluation

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — standardized LLM benchmarks
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) — prime-rl's environment abstraction
