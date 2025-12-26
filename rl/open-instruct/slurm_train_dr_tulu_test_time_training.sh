#!/bin/bash
#SBATCH --job-name=ttt-dr-tulu
#SBATCH --account=rulins
#SBATCH --qos=normal
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/scrubbed/rulins/slurm_logs/dr-tulu/ttt/slurm-%A_%a.out


# Change to the working directory
cd /gpfs/projects/kohlab/rulins/dr-tulu/rl/open-instruct

# Run the training script
bash train_dr_tulu_test_time_training.sh

