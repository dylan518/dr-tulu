#!/bin/bash
#SBATCH --job-name=eval-browsecomp
#SBATCH --account=rulins
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scrubbed/rulins/slurm_logs/dr-tulu/eval/browsecomp-%j.out

# Generic BrowseComp evaluation script
# Expects environment variables: STEP, CKPT_DIR, NUM_EXAMPLES

set -e

# Defaults if not set
STEP="${STEP:-100}"
NUM_EXAMPLES="${NUM_EXAMPLES:-100}"

# Validate checkpoint directory
if [ -z "$CKPT_DIR" ] || [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: CKPT_DIR not set or directory not found: $CKPT_DIR"
    exit 1
fi

DATASET="browsecomp"
EVAL_OUTPUT_DIR="/gpfs/scrubbed/rulins/dr-tulu/eval_output/dr-tulu-shortform-${DATASET}-step${STEP}-n${NUM_EXAMPLES}"

# Server ports
MODEL_PORT=30001
BROWSE_MODEL_PORT=30002
MCP_PORT=8000
MAX_CONCURRENT=20

echo "=============================================="
echo "Evaluating checkpoint on $DATASET (n=$NUM_EXAMPLES)"
echo "=============================================="
echo "Checkpoint: $CKPT_DIR"
echo "Step:       $STEP"
echo "Samples:    $NUM_EXAMPLES"
echo "Output dir: $EVAL_OUTPUT_DIR"
echo "=============================================="

# ============================================
# Activate environment
# ============================================
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

eval "$(conda shell.bash hook)"
conda activate /gpfs/projects/kohlab/rulins/env/dr_agent

cd /gpfs/projects/kohlab/rulins/dr-tulu/agent

# Source environment variables (API keys)
export $(grep -v '^#' .env | xargs)

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
screen -dmS vllm_main bash -c "eval \"\$(conda shell.bash hook)\" && conda activate /gpfs/projects/kohlab/rulins/env/dr_agent && CUDA_VISIBLE_DEVICES=0 vllm serve $CKPT_DIR --dtype auto --port $MODEL_PORT --max-model-len 40960 2>&1 | tee /tmp/vllm_main.log"

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
# Run evaluation (generation)
# ============================================
echo "=============================================="
echo "Running $DATASET generation (n=$NUM_EXAMPLES)..."
echo "=============================================="

mkdir -p "$EVAL_OUTPUT_DIR"

python workflows/auto_search_sft.py \
    generate-dataset $DATASET \
    --num-examples $NUM_EXAMPLES \
    --max-concurrent $MAX_CONCURRENT \
    --batch-size $MAX_CONCURRENT \
    --use-cache \
    --config workflows/auto_search_sft.yaml \
    --config-overrides "search_agent_model_name=$CKPT_DIR,use_browse_agent=true,search_agent_max_tool_calls=20,browse_tool_name=jina" \
    --output "$EVAL_OUTPUT_DIR/${DATASET}.jsonl"

# ============================================
# Run scoring
# ============================================
echo "=============================================="
echo "Running BrowseComp evaluation..."
echo "=============================================="

python scripts/evaluate.py $DATASET "$EVAL_OUTPUT_DIR/${DATASET}.jsonl" --grader-model gpt-4o

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $EVAL_OUTPUT_DIR"
echo "=============================================="

# Cleanup
screen -S vllm_main -X quit 2>/dev/null || true
screen -S vllm_browse -X quit 2>/dev/null || true
screen -S mcp_server -X quit 2>/dev/null || true

