#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:3
#SBATCH --job-name=mnli
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --exclude=materialgpu00

nvidia-smi

python run_glue.py configs/MNLI/base.json
