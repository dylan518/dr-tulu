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
# This should be >= GLM47_MAX_TOKENS used by the agent workflow.
#
# NOTE: Increasing MAX_MODEL_LEN increases KV cache requirements and may reduce throughput or OOM
# depending on GPU_MEMORY_UTILIZATION. If you see OOMs at startup, drop this back to 8192.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# Must be high enough that vLLM has non-negative KV cache memory after weights load.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

TP="${TP:-8}"

# Throughput knobs:
# - MAX_NUM_SEQS: max concurrent sequences the scheduler can run.
# - MAX_NUM_BATCHED_TOKENS: upper bound on total tokens per batch (prefill + decode).
# These defaults are conservative; increase for stress-testing if you have KV headroom.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
# IMPORTANT (stability):
# Overriding `--max-num-batched-tokens` can create very large/irregular batch shapes that
# stress fused kernels (especially FP8 MoE). If you leave it unset, vLLM will choose a
# default based on model + GPU constraints.
#
# To use vLLM defaults, set this env var to an empty string:
#   MAX_NUM_BATCHED_TOKENS=""
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS-}"

# Tool calling / parsing knobs:
# vLLM's GLM tool parser can be fragile if the model emits malformed tool-call text.
# You can disable all tool parsing (recommended for stability) and use the repo's
# parser-based tool calling instead by setting:
#   VLLM_ENABLE_AUTO_TOOL_CHOICE=0 VLLM_TOOL_CALL_PARSER="" VLLM_REASONING_PARSER=""
# NOTE: use "-" (not ":-") so that explicitly setting an empty string disables the flag.
# - ${VAR-default} uses default only when VAR is unset
# - ${VAR:-default} uses default when VAR is unset OR empty
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE-1}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER-glm45}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER-glm45}"

cmd=(uv run vllm serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --port "$PORT"
  --dtype auto
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --tensor-parallel-size "$TP"
  --enforce-eager
)

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "${VLLM_ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  cmd+=(--enable-auto-tool-choice)
fi
if [[ -n "${VLLM_TOOL_CALL_PARSER}" ]]; then
  cmd+=(--tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
fi
if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
  cmd+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
fi

exec "${cmd[@]}"
