import os
import json
import glob
import subprocess

header = '''#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --job-name=sstorig
#SBATCH --time=0-02:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --exclude=materialgpu00

nvidia-smi

'''

task_name = 'SST-2-GLUE'
must_have_substrings = ['dev-']

for i, filename in enumerate(glob.iglob(f'configs/{task_name}/**/*.json', recursive=True)):
    args = json.load(open(filename))

    checkpoint_dir = os.path.join(args['output_dir'], 'pytorch_model.bin')
    if os.path.exists(checkpoint_dir):
        continue

    if not all(x in filename for x in must_have_substrings):
        continue

    print(filename)
    with open('run.slurm', 'w') as f:
        f.write(header)
        f.write('python run_glue.py {}'.format(filename))
    subprocess.run(['sbatch', 'run.slurm'])
