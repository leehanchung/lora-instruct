#!/bin/bash
# Serve Qwen3.5-2B-Base via SGLang as an OpenAI-compatible endpoint.
#
# SGLang is slime's RL rollout engine, so eval and RL training share one
# inference stack. OpenAI-compatible API at http://<host>:<port>/v1.
#
#   --tool-call-parser is REQUIRED for the pi agent (OpenAI tool_choice=auto);
#     `qwen25` matches Qwen's hermes-style <tool_call> format.
set -euo pipefail

VENV="${SGLANG_VENV:-$HOME/venvs/sglang-qwen}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
SERVED_NAME="${SERVED_NAME:-qwen35-2b-base}"
PORT="${PORT:-30000}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
# CuDNN check aborts on a PyTorch 2.9.1 Conv3d/vision bug irrelevant to this
# text-only model; skip it so startup proceeds.
export SGLANG_DISABLE_CUDNN_CHECK=1

# Triton attention + PyTorch sampling avoid flashinfer's nvcc JIT, which would
# otherwise compile sm_89 kernels with the system CUDA 11.7 toolkit (too old for
# the RTX 4090) and abort. Triton compiles via LLVM — no CUDA toolkit needed.
exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --context-length "${CONTEXT_LEN:-16384}" \
  --mem-fraction-static "${MEM_FRACTION:-0.85}" \
  --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" \
  --attention-backend "${ATTN_BACKEND:-triton}" \
  --sampling-backend "${SAMPLING_BACKEND:-pytorch}"
