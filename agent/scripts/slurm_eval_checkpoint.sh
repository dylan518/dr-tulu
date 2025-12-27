#!/bin/bash
#SBATCH --job-name=eval-dr-tulu
#SBATCH --account=rulins
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/rulins/slurm_logs/dr-tulu/eval/slurm-%A_%a.out

# Create log directory
mkdir -p /gpfs/scrubbed/rulins/slurm_logs/dr-tulu/eval

# Configuration - modify STEP as needed
STEP="${1:-115}"

echo "=============================================="
echo "SLURM Evaluation Job"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Step:       $STEP"
echo "=============================================="

# Change to the working directory
cd /gpfs/projects/kohlab/rulins/dr-tulu/agent

# Run the evaluation script
bash scripts/eval_local_checkpoint.sh $STEP

