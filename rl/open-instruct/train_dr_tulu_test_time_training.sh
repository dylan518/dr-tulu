model_path=rl-research/DR-Tulu-8B
dataset_list="rl-research/healthbench_all_adaptive_rubric 1.0"
exp_name="dr-tulu-ttt-1node"
# Test-time training: Only use adaptive rubrics (no static rubrics from dataset)

# if using the docker container, you can use this path. Otherwise, you need to set the path to the blocklist file.
# export CRAWL4AI_BLOCKLIST_PATH=/gpfs/projects/kohlab/rulins/dr-tulu/rl/open-instruct/crawl4ai_block_list.txt
export MCP_MAX_CONCURRENT_CALLS=512
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RUBRIC_JUDGE_MODEL=gpt-4.1-mini
export MCP_CACHE_DIR=.cache-${RANDOM}
export MCP_TRANSPORT_PORT=8003

export OUTPUT_DIR=/gpfs/scrubbed/rulins/dr-tulu/

uv run --extra compile python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --wandb_project_name rl-rag \
        --beta 0.001 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 32 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 5e-7 \
        --per_device_train_batch_size 1 \
        --output_dir $OUTPUT_DIR/output \
        --kl_estimator kl3 \
        --dataset_mixer_list ${dataset_list} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list rl-research/healthbench_all_adaptive_rubric 32 \
        --dataset_mixer_eval_list_splits test \
        --apply_adaptive_rubric_reward true \
        --normalize_rubric_scores false \
        --use_rubric_buffer true \
        --use_static_rubrics_as_persistent_rubrics false \
        --only_adaptive_rubrics_for_training true \
        --max_active_rubrics 5 \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 16384 \
        --pack_length 18500 \
        --model_name_or_path ${model_path} \
        --non_stop_penalty False \
        --non_stop_penalty_value 0.0 \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 4 \
        --vllm_num_engines 4 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --no_citation_reward true \
        --seed 1 \
        --num_evals 500 \
        --save_freq 5 \
        --try_launch_beaker_eval_jobs_on_weka False \
        --gradient_checkpointing \
        --with_tracking \
        --max_tool_calls 10 \
        --only_reward_good_outputs False \
        --tools mcp \
        --checkpoint_state_freq 5 \
        --checkpoint_state_dir $OUTPUT_DIR/output/checkpoints/${exp_name} \
        --keep_last_n_checkpoints 3 \
        --mcp_parser_name v20250824 \
        --system_prompt_file open_instruct/search_utils/system_prompts/unified_tool_calling_v20250907.yaml  \
        --mcp_tool_names 'snippet_search,google_search,browse_webpage' \
        --mcp_server_command "uv run python -m dr_agent.mcp_backend.main --transport http --port 8003 --host 0.0.0.0 --path /mcp"

