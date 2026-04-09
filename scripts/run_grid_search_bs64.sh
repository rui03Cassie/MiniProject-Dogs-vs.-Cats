#!/usr/bin/env bash
PYTHONPATH=. python tools/run_grid_search.py \
  --data_root ./data/datasets \
  --output_dir outputs/grid_search_fixed_ep30_bs64 \
  --models cnn resnet \
  --augmentations none light strong \
  --lrs 0.001 0.0005 0.0001 \
  --batch_sizes 64 \
  --optimizers adam adamw \
  --schedulers plateau cosine \
  --epochs_list 50 \
  --early_stop_patience 10
