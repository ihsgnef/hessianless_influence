#!/bin/bash
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=snli
#SBATCH --time=0-08:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --output=/fs/clip-scratch/shifeng/influenceless
#SBATCH --exclude=materialgpu00

python run_glue.py configs/SNLI/random_10_percent_removed_combined/0.json