#!/bin/bash
#SBATCH --job-name=cat_dog_grid_search
#SBATCH --output=/projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs/grid_search_bz128_%j.out
#SBATCH --error=/projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs/grid_search_bz128_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G 
#SBATCH --time=8:00:00

mkdir -p /projects/projectsLC/MiniProject-Dogs-vs.-Cats/logs

module load Miniforge3
source activate ee6483

cd /projects/projectsLC/MiniProject-Dogs-vs.-Cats

# PYTHONPATH=. python tools/run_grid_search.py \
#   --data_root ./data/datasets \
#   --output_dir outputs/grid_search_fixed_ep50_bs128 \
#   --models resnet \
#   --augmentations light strong \
#   --lrs 0.001 0.0005 0.0001 0.00005 \
#   --batch_sizes 128 \
#   --optimizers adam adamw \
#   --schedulers plateau cosine \
#   --epochs_list 50 \
#   --early_stop_patience 10
PYTHONPATH=. python tools/run_grid_search.py \
  --data_root ./data/datasets \
  --output_dir outputs/grid_search_fixed_ep50_bs128 \
  --models resnet \
  --augmentations none \
  --lrs 0.001 0.0005 0.0001 0.00005 \
  --batch_sizes 128 \
  --optimizers adam adamw \
  --schedulers plateau cosine \
  --epochs_list 50 \
  --early_stop_patience 10