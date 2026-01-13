#!/usr/bin/env bash
set -euo pipefail

# Reproducible vLLM launch for GPT-OSS 120B (OpenAI-compatible server).
#
# Usage:
#   cd /home/ubuntu/dr-tulu/agent
#   set -a; source /home/ubuntu/dr-tulu/rl/open-instruct/.env; set +a
#   HF_TOKEN="$HF_KEY2" ./scripts/serve_gpt_oss_120b_vllm.sh
#
# Notes:
# - Uses all 8 GPUs via TP=8.
# - Uses --enforce-eager to avoid long torch.compile startup times and memory spikes.
# - Exposes the model under `gpt-oss-120b` by default on port 30001 (matches LiteLLM's openai/ prefix rewrite).

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL="${MODEL:-openai/gpt-oss-120b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-120b}"
PORT="${PORT:-30001}"

# Keep context modest for stability; raise if you know you have enough KV cache headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# Must be high enough that vLLM has non-negative KV cache memory after weights load.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

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
  --tool-call-parser openai


