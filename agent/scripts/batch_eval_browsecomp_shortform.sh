#!/bin/bash
# Batch submit BrowseComp evaluations for shortform checkpoints
# Usage: bash scripts/batch_eval_browsecomp_shortform.sh

set -e

# Create log directory
mkdir -p /gpfs/scrubbed/rulins/slurm_logs/dr-tulu/eval

# Mapping: step -> checkpoint directory
declare -A CKPT_MAP
CKPT_MAP[10]="/gpfs/scrubbed/rulins/dr-tulu/output/dr-tulu-rl-shortform__1__1766609175_checkpoints"
CKPT_MAP[50]="/gpfs/scrubbed/rulins/dr-tulu/output/dr-tulu-rl-shortform__1__1766745473_checkpoints"
CKPT_MAP[100]="/gpfs/scrubbed/rulins/dr-tulu/output/dr-tulu-rl-shortform__1__1766833709_checkpoints"
CKPT_MAP[150]="/gpfs/scrubbed/rulins/dr-tulu/output/dr-tulu-rl-shortform__1__1766977546_checkpoints"
CKPT_MAP[200]="/gpfs/scrubbed/rulins/dr-tulu/output/dr-tulu-rl-shortform__1__1766977546_checkpoints"

NUM_EXAMPLES=100

for STEP in 10 50 100 150 200; do
    CKPT_BASE="${CKPT_MAP[$STEP]}"
    CKPT_DIR="${CKPT_BASE}/step_${STEP}"
    
    if [ ! -d "$CKPT_DIR" ]; then
        echo "WARNING: Checkpoint not found: $CKPT_DIR"
        continue
    fi
    
    echo "Submitting job for step $STEP (checkpoint: $CKPT_DIR)"
    
    sbatch --job-name="bc-s${STEP}" \
           --output="/gpfs/scrubbed/rulins/slurm_logs/dr-tulu/eval/browsecomp-shortform-step${STEP}-%j.out" \
           --export=ALL,STEP=$STEP,CKPT_DIR=$CKPT_DIR,NUM_EXAMPLES=$NUM_EXAMPLES \
           /gpfs/projects/kohlab/rulins/dr-tulu/agent/scripts/slurm_eval_browsecomp_generic.sh
    
    # Small delay between submissions
    sleep 2
done

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"

