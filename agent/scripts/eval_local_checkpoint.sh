#!/bin/bash
# Evaluate a local HuggingFace checkpoint on HealthBench
# Usage: bash scripts/eval_local_checkpoint.sh [STEP]
# Example: bash scripts/eval_local_checkpoint.sh 115

set -e

# ============================================
# Configuration
# ============================================
STEP="${1:-115}"
DATASET="healthbench"
CHECKPOINT_BASE="/gpfs/scrubbed/rulins/dr-tulu/output/dr-tulu-ttt-1node__1__1766744848_checkpoints"
HF_CHECKPOINT_DIR="${CHECKPOINT_BASE}/step_${STEP}"
EVAL_OUTPUT_DIR="/gpfs/scrubbed/rulins/dr-tulu/eval_output/dr-tulu-ttt-${DATASET}-step${STEP}"

# Server ports
MODEL_PORT=30001
BROWSE_MODEL_PORT=30002
MCP_PORT=8000
MAX_CONCURRENT=20

# ============================================
# Validate checkpoint
# ============================================
if [ ! -d "$HF_CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $HF_CHECKPOINT_DIR"
    echo "Available checkpoints:"
    ls -1 "$CHECKPOINT_BASE" | grep step_ | sort -t_ -k2 -n
    exit 1
fi

echo "=============================================="
echo "Evaluating checkpoint on $DATASET"
echo "=============================================="
echo "Checkpoint: $HF_CHECKPOINT_DIR"
echo "Step:       $STEP"
echo "Output dir: $EVAL_OUTPUT_DIR"
echo "=============================================="

# ============================================
# Activate environment
# ============================================
# Deactivate any existing conda env first to avoid conflicts
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

# Activate the dr_agent environment
eval "$(conda shell.bash hook)"
conda activate /gpfs/projects/kohlab/rulins/env/dr_agent

cd /gpfs/projects/kohlab/rulins/dr-tulu/agent

# ============================================
# Kill existing servers
# ============================================
echo "Cleaning up existing servers..."
pkill -f "vllm serve.*:${MODEL_PORT}" 2>/dev/null || true
pkill -f "vllm serve.*:${BROWSE_MODEL_PORT}" 2>/dev/null || true
pkill -f "mcp_backend.*:${MCP_PORT}" 2>/dev/null || true
screen -S vllm_main -X quit 2>/dev/null || true
screen -S vllm_browse -X quit 2>/dev/null || true
screen -S mcp_server -X quit 2>/dev/null || true
sleep 2

# ============================================
# Launch VLLM server (main model on GPU 0)
# ============================================
echo "Starting main VLLM server on port $MODEL_PORT (GPU 0)..."
screen -dmS vllm_main bash -c "eval \"\$(conda shell.bash hook)\" && conda activate /gpfs/projects/kohlab/rulins/env/dr_agent && CUDA_VISIBLE_DEVICES=0 vllm serve $HF_CHECKPOINT_DIR --dtype auto --port $MODEL_PORT --max-model-len 40960 2>&1 | tee /tmp/vllm_main.log"

# ============================================
# Launch VLLM browse agent server (Qwen3-8B on GPU 1)
# ============================================
echo "Starting browse agent VLLM server on port $BROWSE_MODEL_PORT (GPU 1)..."
screen -dmS vllm_browse bash -c "eval \"\$(conda shell.bash hook)\" && conda activate /gpfs/projects/kohlab/rulins/env/dr_agent && CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port $BROWSE_MODEL_PORT --max-model-len 40960 2>&1 | tee /tmp/vllm_browse.log"

# ============================================
# Launch MCP server
# ============================================
echo "Starting MCP server on port $MCP_PORT..."
screen -dmS mcp_server bash -c "eval \"\$(conda shell.bash hook)\" && conda activate /gpfs/projects/kohlab/rulins/env/dr_agent && cd /gpfs/projects/kohlab/rulins/dr-tulu/agent && python -m dr_agent.mcp_backend.main --port $MCP_PORT 2>&1 | tee /tmp/mcp_server.log"

# ============================================
# Wait for servers to be ready
# ============================================
echo "Waiting for main VLLM server to start..."
for i in {1..120}; do
    if curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1; then
        echo "Main VLLM server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Main VLLM server failed to start. Check /tmp/vllm_main.log"
        exit 1
    fi
    sleep 5
done

echo "Waiting for browse agent VLLM server to start..."
for i in {1..120}; do
    if curl -s http://localhost:$BROWSE_MODEL_PORT/health > /dev/null 2>&1; then
        echo "Browse agent VLLM server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Browse agent VLLM server failed to start. Check /tmp/vllm_browse.log"
        exit 1
    fi
    sleep 5
done

# ============================================
# Run evaluation
# ============================================
echo "=============================================="
echo "Running $DATASET evaluation..."
echo "=============================================="

mkdir -p "$EVAL_OUTPUT_DIR"

python workflows/auto_search_sft.py \
    generate-dataset $DATASET \
    --num-examples final_run \
    --max-concurrent $MAX_CONCURRENT \
    --batch-size $MAX_CONCURRENT \
    --use-cache \
    --config workflows/auto_search_sft.yaml \
    --config-overrides "search_agent_model_name=$HF_CHECKPOINT_DIR,use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=jina" \
    --output "$EVAL_OUTPUT_DIR/${DATASET}.jsonl"

# ============================================
# Run scoring
# ============================================
echo "=============================================="
echo "Running evaluation metrics..."
echo "=============================================="

# Use dedicated healthbench evaluator (handles deduplication, uses official scoring)
python scripts/evaluate_healthbench.py "$EVAL_OUTPUT_DIR/${DATASET}.jsonl"

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $EVAL_OUTPUT_DIR"
echo "=============================================="
echo ""
echo "To cleanup servers, run:"
echo "  screen -S vllm_main -X quit"
echo "  screen -S vllm_browse -X quit"
echo "  screen -S mcp_server -X quit"

