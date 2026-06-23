#!/usr/bin/env bash
# Launch a slime deep-research RL run.
#
# slime is the engine (Docker image); this script only assembles the three arg
# blocks (Megatron / SGLang / slime) and points slime at OUR plugin functions.
# Hyperparameters come from configs/ — keep them there, not here.
#
# Usage:
#   bash launch/run.sh                       # uses configs/base.yaml
#   EXPERIMENT=grpo_glm_8b bash launch/run.sh
#
# Expected to run INSIDE the slimerl/slime container with this recipe mounted at
# /workspace and the search_server reachable at $TOOL_SERVER_URL.
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${RECIPE_DIR}:/root/Megatron-LM:${PYTHONPATH:-}"

# --- Megatron model args (convert HF -> dist first; see README) -------------
MODEL_ARGS=(
  --hf-checkpoint /root/GLM-Z1-9B-0414
  --tensor-model-parallel-size 2
)

# --- SGLang rollout args -----------------------------------------------------
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
)

# --- slime RL args: wire in OUR plugins (no engine fork) ---------------------
SLIME_ARGS=(
  --custom-generate-function-path plugins.rollout.generate
  --custom-rm-path                plugins.reward.reward_func
  --n-samples-per-prompt 8
  --global-batch-size 256
  --lr 1e-6
)

python3 train.py "${MODEL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${SLIME_ARGS[@]}"
