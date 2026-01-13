#!/usr/bin/env bash
set -euo pipefail

# Reproducible vLLM launch for GLM-4.7-FP8 (OpenAI-compatible server).
#
# This script assumes you're running from within the `agent/` directory:
#   cd /home/ubuntu/dr-tulu/agent
#
# Notes:
# - We use all 8 GPUs via TP=8.
# - We use `--enforce-eager` to avoid long torch.compile startup times and memory spikes.
# - Tool calling is enabled for GLM via vLLM's tool-call parser and auto tool choice.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL="${MODEL:-zai-org/GLM-4.7-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
PORT="${PORT:-30002}"

# Keep context modest for stability; raise if you know you have enough KV cache headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Must be high enough that vLLM has non-negative KV cache memory after weights load.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

TP="${TP:-8}"

exec uv run vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TP" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser glm45 \
  --reasoning-parser glm45


