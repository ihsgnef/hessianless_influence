import glob
import subprocess

header = '''#!/bin/bash
#SBATCH --qos=gpu-short
#SBATCH --partition=gpu
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=sst
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --output=/fs/clip-scratch/shifeng/influenceless/logs/
#SBATCH --excluds=materialgpu00

'''


max_jobs = 100
for i, filename in enumerate(glob.iglob('configs/**/*.json', recursive=True)):
    if i > max_jobs:
        break
    with open('run.slurm', 'w') as f:
        f.write(header)
        f.write('python run_glue.py {}'.format(filename))
    subprocess.run(['sbatch', 'run.slurm'])
