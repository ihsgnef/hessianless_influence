import os
import json
import glob
import subprocess

header = '''#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --job-name={}
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20g
#SBATCH --exclude=materialgpu00

nvidia-smi

'''

header = '''#!/bin/bash
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --job-name={}
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/fs/clip-scratch/shifeng/influenceless
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20g
#SBATCH --exclude=materialgpu00

nvidia-smi

'''

task_name = 'SNLI'
must_have_substrings = []

for i, filename in enumerate(glob.iglob(f'configs/{task_name}/**/*.json', recursive=True)):
    args = json.load(open(filename))

    checkpoint_dir = os.path.join(args['output_dir'], 'pytorch_model.bin')
    if os.path.exists(checkpoint_dir):
        continue

    if len(must_have_substrings) > 0:
        if not all(x in filename for x in must_have_substrings):
            continue

    print(filename)
    config_name = os.path.basename(filename)
    with open('run.slurm', 'w') as f:
        f.write(header.format(config_name))
        f.write('python run_glue.py {}'.format(filename))
    subprocess.run(['sbatch', 'run.slurm'])
