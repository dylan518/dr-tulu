#!/usr/bin/env bash
set -euo pipefail

# End-to-end reproducible run (OSS 120B):
# 1) Generate SQA/SQA-like long-form answers with the GPT-OSS-120B vLLM-backed workflow.
# 2) Judge the outputs with GPT-4.1-mini.
#
# This script is resumable: if the output JSONL already exists, `--use-cache` will skip completed IDs.
#
# Required env:
# - HF_TOKEN (or HF_KEY2/HF_KEY via your .env): to download the AstaBench file
# - SCHOLARQA_CS2_HF_DATASET=allenai/asta-bench
# - SCHOLARQA_CS2_HF_CONFIG=tasks/sqa/rubrics_v2_recomputed.json
# - GPT_OSS_BASE_URL=http://127.0.0.1:30001/v1
# - GPT_OSS_MODEL_NAME=openai/gpt-oss-120b
# - OPENAI_API_KEY=... (for the judge)
#
# Optional:
# - OUT_DIR=eval_output/sqa100_gpt_oss_120b
# - NUM_EXAMPLES=25
# - MAX_CONCURRENT=4
# - BATCH_SIZE=8

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-eval_output/sqa100_gpt_oss_120b}"
NUM_EXAMPLES="${NUM_EXAMPLES:-25}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "$OUT_DIR"

GEN_PATH="$OUT_DIR/sqa100_gpt_oss_120b.jsonl"

echo "[1/2] Generating SQA (AstaBench) responses -> $GEN_PATH"
if [[ ! -f "$GEN_PATH" ]]; then
  : > "$GEN_PATH"
fi

uv run python workflows/auto_search_sft.py generate-dataset scholarqa_cs2 \
  --num-examples "$NUM_EXAMPLES" \
  --use-cache \
  --batch-size "$BATCH_SIZE" \
  --max-concurrent "$MAX_CONCURRENT" \
  --config workflows/auto_search_sft-gpt-oss.yaml \
  --output "$GEN_PATH"

if [[ ! -s "$GEN_PATH" ]]; then
  echo "[ERROR] Generation produced 0 rows (empty $GEN_PATH). Not running judge."
  exit 1
fi

echo "[2/2] Judging with GPT-4.1-mini"
uv run python scripts/evaluate.py scholarqa_cs2 "$GEN_PATH" --grader-model gpt-4.1-mini

echo "[DONE] Outputs:"
echo "  - Generation: $GEN_PATH"
echo "  - Judge JSON: ${GEN_PATH%.jsonl}_eval_results.json"


