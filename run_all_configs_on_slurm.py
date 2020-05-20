import glob
import subprocess

header = '''#!/bin/bash
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=snli
#SBATCH --time=0-08:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --output=/fs/clip-scratch/shifeng/influenceless
#SBATCH --exclude=materialgpu00

'''


for i, filename in enumerate(glob.iglob('configs/SNLI/**/*.json', recursive=True)):
    if 'base' in filename:
        continue
    print(filename)
    with open('run.slurm', 'w') as f:
        f.write(header)
        f.write('python run_glue.py {}'.format(filename))
    subprocess.run(['sbatch', 'run.slurm'])
