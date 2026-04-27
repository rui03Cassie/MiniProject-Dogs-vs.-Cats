#!/bin/bash
#SBATCH --job-name=cifar10_eval
#SBATCH --output=/projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs/cifar10_eval_%j.out
#SBATCH --error=/projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs/cifar10_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00


module load Miniforge3
source activate ee6483

cd /projects/projectsLC/MiniProject-Dogs-vs.-Cats

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time:   $(date)"

export PYTHONPATH=/projects/projectsLC/MiniProject-Dogs-vs.-Cats

python cifar10/eval_cifar10.py \
    --output_dir /projects/projectsLC/MiniProject-Dogs-vs.-Cats/outputs/cifar10 \
    --data_root  /projects/projectsLC/MiniProject-Dogs-vs.-Cats/cifar10_data \
    --img_size   32 \
    --batch_size 256

echo "All experiments done: $(date)"
