#!/bin/bash
#SBATCH --qos=gpu-short
#SBATCH --partition=gpu
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=sst2_random_10_percent_removed_negative_0
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --chdir=/fs/clip-quiz/shifeng/influenceless

zsh
hostname
nvidia-smi
conda activate v2.9.0
python run_glue.py configs/sst2_random_10_percent_removed_negative_0.json
