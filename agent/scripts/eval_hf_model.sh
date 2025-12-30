#!/bin/bash
# Evaluate a HuggingFace model on HealthBench
# Usage: bash scripts/eval_hf_model.sh <model_name> <output_name>
# Example: bash scripts/eval_hf_model.sh rl-research/DR-Tulu-8B dr-tulu-8b-base

set -e

# ============================================
# Configuration
# ============================================
MODEL_NAME="${1:-rl-research/DR-Tulu-8B}"
OUTPUT_NAME="${2:-$(echo $MODEL_NAME | tr '/' '-')}"
DATASET="healthbench"
EVAL_OUTPUT_DIR="/gpfs/scrubbed/rulins/dr-tulu/eval_output/${OUTPUT_NAME}-${DATASET}"

# Server ports
MODEL_PORT=30001
BROWSE_MODEL_PORT=30002
MCP_PORT=8000
MAX_CONCURRENT=20

echo "=============================================="
echo "Evaluating HuggingFace model on $DATASET"
echo "=============================================="
echo "Model:      $MODEL_NAME"
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
# Launch VLLM servers
# ============================================
echo "Starting main VLLM server on port $MODEL_PORT (GPU 0)..."
screen -dmS vllm_main bash -c "source ~/.bashrc && conda activate /gpfs/projects/kohlab/rulins/env/dr_agent && CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME --dtype auto --port $MODEL_PORT --max-model-len 40960 --gpu-memory-utilization 0.45 2>&1 | tee /tmp/vllm_main.log"

echo "Starting browse agent VLLM server on port $BROWSE_MODEL_PORT (GPU 1)..."
screen -dmS vllm_browse bash -c "source ~/.bashrc && conda activate /gpfs/projects/kohlab/rulins/env/dr_agent && CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port $BROWSE_MODEL_PORT --max-model-len 40960 --gpu-memory-utilization 0.45 2>&1 | tee /tmp/vllm_browse.log"

# ============================================
# Launch MCP server
# ============================================
echo "Starting MCP server on port $MCP_PORT..."
screen -dmS mcp_server bash -c "source ~/.bashrc && conda activate /gpfs/projects/kohlab/rulins/env/dr_agent && cd /gpfs/projects/kohlab/rulins/dr-tulu/agent && python -m dr_agent.mcp_backend.main --port $MCP_PORT 2>&1 | tee /tmp/mcp_server.log"

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
    --config-overrides "search_agent_model_name=$MODEL_NAME,use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=jina" \
    --output "$EVAL_OUTPUT_DIR/${DATASET}.jsonl"

# ============================================
# Run scoring
# ============================================
echo "=============================================="
echo "Running evaluation metrics..."
echo "=============================================="

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

