#!/usr/bin/env bash
set -euo pipefail

# End-to-end reproducible run:
# 1) Generate ScholarQA-CS2 answers with the GLM-4.7 vLLM-backed workflow.
# 2) Judge the outputs with GPT-4.1-mini.
#
# Required env:
# - SCHOLARQA_CS2_PATH=/path/to/scholarqa_cs2.jsonl
# - GLM47_BASE_URL=http://127.0.0.1:30002/v1
# - GLM47_MODEL_NAME=zai-org/GLM-4.7-FP8
# - OPENAI_API_KEY=... (for the judge)
#
# Optional:
# - OUT_DIR=eval_output/scholarqa_cs2_glm47
# - NUM_EXAMPLES=10

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-eval_output/scholarqa_cs2_glm47}"
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
BATCH_SIZE="${BATCH_SIZE:-5}"

mkdir -p "$OUT_DIR"

GEN_PATH="$OUT_DIR/scholarqa_cs2_glm47.jsonl"

echo "[1/2] Generating ScholarQA-CS2 responses -> $GEN_PATH"
# Ensure the output file exists even if generation fails early, but do NOT truncate it.
# (Keeping it enables resume/queue behavior via --use-cache.)
if [[ ! -f "$GEN_PATH" ]]; then
  : > "$GEN_PATH"
fi

uv run python workflows/auto_search_sft.py generate-dataset scholarqa_cs2 \
  --num-examples "$NUM_EXAMPLES" \
  --use-cache \
  --batch-size "$BATCH_SIZE" \
  --max-concurrent "$MAX_CONCURRENT" \
  --config workflows/auto_search_sft-glm47.yaml \
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


