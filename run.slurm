#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --job-name=snli
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --exclude=materialgpu00

nvidia-smi

python run_glue.py configs/SNLI/random_10_percent_removed_combined/0.json