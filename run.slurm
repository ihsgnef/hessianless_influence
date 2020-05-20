#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --time=0-8:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=SNLI
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless

python run_glue.py configs/SNLI/base.json
