#!/bin/bash
# Serve Qwen3.5-2B-Base via vLLM as an OpenAI-compatible endpoint on :8000.
#
# Notes:
#   - VLLM_USE_FLASHINFER_SAMPLER=0 uses vLLM's native top-k/top-p sampler and
#     avoids flashinfer's JIT (which needs `ninja` + nvcc on PATH).
#   - venv/bin is prepended to PATH so any remaining JIT path can find `ninja`.
#   - --served-model-name is the alias pi/benchflow target (vllm/qwen35-2b-base).
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/venvs/vllm-qwen}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
SERVED_NAME="${SERVED_NAME:-qwen35-2b-base}"
PORT="${PORT:-8000}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES="$GPU"

# --enable-auto-tool-choice + --tool-call-parser are REQUIRED for the pi agent:
# pi sends OpenAI-native tool definitions with tool_choice=auto, which vLLM
# rejects with HTTP 400 unless a parser is configured. `hermes` is Qwen's
# tool-call format. (TOOL_PARSER overridable: hermes | qwen3xml | qwen3coder.)
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-hermes}"
