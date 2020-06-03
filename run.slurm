#!/bin/bash
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:4
#SBATCH --job-name=least_dot_gradient_similar_10_percent_to_combined_dev-5234_removed.json
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20g
#SBATCH --exclude=materialgpu00

nvidia-smi

python snli_exp.py
