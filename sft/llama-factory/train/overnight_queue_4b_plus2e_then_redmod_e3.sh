#!/usr/bin/env bash
set -euo pipefail

# Queue two runs overnight (no parallelism). Run this from the repo root or anywhere.
# It will cd into sft/llama-factory so relative paths inside configs work.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p train/logs

ts="$(date +%Y%m%d_%H%M%S)"

echo "=== Starting overnight queue at $(date) ===" | tee "train/logs/overnight_queue.${ts}.log"

echo "=== RUN 1: Fresh 4B SFT (paper-style span masking) ===" | tee -a "train/logs/overnight_queue.${ts}.log"
bash train/train.sh train/qwen3-4B-thinking-2507-full-sft-masked-paper.yaml \
  2>&1 | tee "train/logs/run1_4b_masked_paper.${ts}.log"

echo "=== RUN 2: Fresh 4B SFT on RedMod/rl_1023_s100_q for 3 epochs ===" | tee -a "train/logs/overnight_queue.${ts}.log"
bash train/train.sh train/qwen3-4B-thinking-2507-full-sft-redmod-rl_1023_s100_q-3epochs.yaml \
  2>&1 | tee "train/logs/run2_redmod_rl_1023_s100_q_e3.${ts}.log"

echo "=== Finished overnight queue at $(date) ===" | tee -a "train/logs/overnight_queue.${ts}.log"


