#!/usr/bin/env bash
set -euo pipefail

# This repo expects you to run via `uv` so dependency versions (and optional compiled deps like flash-attn)
# match what DR-Tulu/Open-Instruct was tested with.
#
# Note: your crash is a NCCL SIGSEGV inside `ncclTopoCheckNet()` (see Ray worker *.err logs).
# On machines without IB / with UCX present, a common workaround is to force NCCL to use sockets.
# You can override any of these externally.
export DR_TULU_DEEPSPEED_DIST_BACKEND=gloo
export NCCL_IB_DISABLE
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export MCP_TRANSPORT_PORT=8003

# LiteLLM (LLM judge) throttling:
# If you see LiteLLM RateLimitError / HTTP 429 / RESOURCE_EXHAUSTED, lower concurrency here.
# (Default inside the code is 256 concurrent calls which can easily trip provider quotas.)
export LITELLM_MAX_CONCURRENT_CALLS="${LITELLM_MAX_CONCURRENT_CALLS:-8}"
# Optional: very noisy but useful to debug provider routing / retries
# export LITELLM_DEBUG=1

output_dir="/home/ubuntu/dr-tulu/rl_runs/dr_tulu_grpo_eval30_1767911717/dr_tulu_grpo_debug_threadstall_eval30__22__1767911726"
mkdir -p "${output_dir}"

# Attention backend:
# - Default is FlashAttention-2 (fastest) via `flash-attn` when running `uv run --extra compile ...`.
# - If you see GPU kernel faults (e.g. NVRM Xid 13 misaligned address / warp errors), try:
#     --attn_implementation sdpa
#   or:
#     --attn_implementation eager
# to confirm whether FlashAttention is the trigger.
uv run --extra compile python -u open_instruct/grpo_fast.py \
  --model_name_or_path /home/ubuntu/dr-tulu/sft/llama-factory/ckpts/qwen3-4B-thinking-2507/full/sft \
  --attn_implementation sdpa \
  --gradient_checkpointing True \
  --use_cache False \
  --dataset_mixer_list rl-research/dr-tulu-rl-data 1.0 \
  --dataset_mixer_eval_list rl-research/dr-tulu-rl-data 30 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list_splits train \
  --dataset_transform_fn rlvr_tokenize_rl_rag_v1 rlvr_filter_v1 \
  --dataset_cache_mode local \
  --dataset_local_cache_dir /home/ubuntu/dr-tulu/rl/open-instruct/local_dataset_cache \
  --max_token_length 18240 \
  --max_prompt_token_length 3000 \
  --system_prompt_file open_instruct/search_utils/system_prompts/unified_tool_calling_v20250907_no_snippet.yaml \
  --exp_name dr_tulu_grpo_debug_threadstall_eval30 \
  --seed 22 \
  --run_name dr_tulu_grpo_debug_threadstall_eval30__22__1767911726 \
  --learning_rate 5e-07 \
  --lr_scheduler_type constant \
  --warm_up_steps 0 \
  --per_device_train_batch_size 1 \
  --total_episodes 400 \
  --world_size 4 \
  --num_evals 1 \
  --save_freq -1 \
  --allow_world_padding \
  --response_length 14000 \
  --temperature 1.0 \
  --num_unique_prompts_rollout 16 \
  --num_samples_per_prompt_rollout 8 \
  --num_samples_per_prompt_eval 8 \
  --beta 0.001 \
  --clip_lower 0.2 \
  --clip_higher 0.2 \
  --kl_estimator kl3 \
  --pack_length 18500 \
  --alpha 0.6 \
  --apply_r1_style_format_reward \
  --r1_style_format_reward 0.2 \
  --additive_format_reward \
  --apply_verifiable_reward \
  --verification_reward 10.0 \
  --verifier_strategy judge \
  --llm_judge_model gemini/gemini-2.5-flash-lite \
  --llm_judge_max_tokens 2048 \
  --llm_judge_temperature 1.0 \
  --llm_judge_timeout 60 \
  --llm_judge_max_context_length 2048 \
  --num_learners_per_node 4 \
  --vllm_num_engines 4 \
  --vllm_tensor_parallel_size 1 \
  --vllm_enforce_eager True \
  --vllm_sync_backend gloo \
  --vllm_sync_timeout_seconds 240 \
  --vllm_gpu_memory_utilization 0.8 \
  --vllm_top_p 0.9 \
  --deepspeed_stage 3 \
  --gather_whole_model True \
  --push_to_hub False \
  --output_dir "${output_dir}" \
  --save_traces True \
  --tools mcp \
  --max_tool_calls 20 \
  --mask_tool_use True \
  --tool_max_concurrency 32 \
  --number_documents_to_search 10 \
  --mcp_tool_names google_search \
  --mcp_parser_name v20250824 \
  --mcp_server_command "uv run python -m dr_agent.mcp_backend.main --transport http --port 8003 --host 0.0.0.0 --path /mcp" \
  --mcp_timeout 180 \
  --context_chars 6000 \
  --overwrite_reward_fn_tag general_rubric \
  --apply_adaptive_rubric_reward True \
  --use_full_responses_for_adaptive_rubric True \
  --use_static_rubrics_as_persistent_rubrics True \
  --add_static_rubrics_to_active_rubrics_every_n_steps 10 \
  --max_active_rubrics 5 \
  --eval_at_step -1 \
  --eval_timeout 1800 \
  --use_full_response_as_answer False
