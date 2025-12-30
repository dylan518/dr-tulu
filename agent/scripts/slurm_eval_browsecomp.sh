#!/bin/bash
#SBATCH --job-name=eval-browsecomp
#SBATCH --account=rulins
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scrubbed/rulins/slurm_logs/dr-tulu/eval/browsecomp-%A_%a.out

set -e

# Arguments: STEP and NUM_EXAMPLES (defaults: 240 and 100)
STEP="${1:-240}"
NUM_EXAMPLES="${2:-100}"

cd /gpfs/projects/kohlab/rulins/dr-tulu/agent

bash scripts/eval_browsecomp.sh $STEP $NUM_EXAMPLES

