#!/bin/bash
#SBATCH --job-name=dogcat_eval
#SBATCH --output=/projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs/dogcat_eval_%j.out
#SBATCH --error=/projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs/dogcat_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

module load Miniforge3
source activate ee6483

cd /projects/projectsLC/MiniProject-Dogs-vs.-Cats
# mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time:   $(date)"

export PYTHONPATH=/projects/projectsLC/MiniProject-Dogs-vs.-Cats

python cifar10/eval_dogcat.py

echo "Done: $(date)"
