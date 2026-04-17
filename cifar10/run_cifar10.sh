#!/bin/bash
#SBATCH --job-name=cifar10_train
#SBATCH --output=/projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs/cifar10_%j.out
#SBATCH --error=/projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs/cifar10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G 
#SBATCH --time=3:00:00
#SBATCH --nodelist=gpu-v100-2

# ── activate the environment ──────────────────
module load Miniforge3
source activate ee6483

cd /projects/projectsLC/MiniProject-Dogs-vs.-Cats

# mkdir -p logs/cifar10

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time:   $(date)"
echo "=============================="

# # ── 实验1: Baseline（无不平衡处理） ────────────────────
# echo ">>> Running Experiment 1: baseline (no imbalance)"
# python cifar10/train_cifar10.py \
#   --model cnn \
#   --epochs 40 \
#   --batch_size 256 \
#   --augmentation light \
#   --imbalance_method none \
#   --lr 1e-3 \
#   --scheduler cosine \
#   --use_amp \
#   --output_dir outputs/cifar10 \
#   --run_name exp1_baseline

# # ── 实验2: 类别权重 ─────────────────────────────────────
# echo ">>> Running Experiment 2: class_weight"
# python cifar10/train_cifar10.py \
#   --model cnn \
#   --epochs 40 \
#   --batch_size 256 \
#   --augmentation light \
#   --imbalance_method class_weight \
#   --lr 1e-3 \
#   --scheduler cosine \
#   --use_amp \
#   --output_dir outputs/cifar10 \
#   --run_name exp2_class_weight

# # ── 实验3: 过采样 ────────────────────────────────────────
# echo ">>> Running Experiment 3: oversample"
# python cifar10/train_cifar10.py \
#   --model cnn \
#   --epochs 40 \
#   --batch_size 256 \
#   --augmentation light \
#   --imbalance_method oversample \
#   --lr 1e-3 \
#   --scheduler cosine \
#   --use_amp \
#   --output_dir outputs/cifar10 \
#   --run_name exp3_oversample

# ── 实验4: 不平衡baseline ────────────────────────────────────────
echo ">>> Running Experiment 4: imbalanced_baseli"
python cifar10/train_cifar10.py \
  --model cnn \
  --epochs 40 \
  --batch_size 256 \
  --augmentation light \
  --imbalance_method imbalanced_only \
  --lr 1e-3 \
  --scheduler cosine \
  --use_amp \
  --output_dir outputs/cifar10 \
  --run_name exp4_imbalanced_baseline

echo "=============================="
echo "All experiments done: $(date)"
echo "=============================="