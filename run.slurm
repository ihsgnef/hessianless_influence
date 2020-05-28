#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --job-name=sstorig
#SBATCH --time=0-02:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --exclude=materialgpu00

nvidia-smi

python run_glue.py configs/SST-2-ORIG/most_dot_gradient_similar_10_percent_to_combined_dev-379_removed.json