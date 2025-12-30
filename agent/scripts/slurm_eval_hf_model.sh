#!/bin/bash
#SBATCH --job-name=eval-hf-model
#SBATCH --account=rulins
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/rulins/slurm_logs/dr-tulu/eval/slurm-%A_%a.out

# Usage: sbatch scripts/slurm_eval_hf_model.sh <model_name> <output_name>
# Example: sbatch scripts/slurm_eval_hf_model.sh rl-research/DR-Tulu-8B dr-tulu-8b-base

MODEL_NAME="${1:-rl-research/DR-Tulu-8B}"
OUTPUT_NAME="${2:-$(echo $MODEL_NAME | tr '/' '-')}"

cd /gpfs/projects/kohlab/rulins/dr-tulu/agent

bash scripts/eval_hf_model.sh "$MODEL_NAME" "$OUTPUT_NAME"

