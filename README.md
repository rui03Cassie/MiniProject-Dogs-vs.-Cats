# MiniProject-Dogs-vs.-Cats
To record Dogs vs. Cats  recognition project 

# EE6483 Mini Project — Dogs vs. Cats & CIFAR-10

Image classification project for EE6483 Artificial Intelligence and Data Mining (NTU). The project covers binary classification (Dogs vs. Cats) and extends to 10-class classification (CIFAR-10) with a study of class imbalance handling strategies.

---

## Dataset

The Dogs vs. Cats dataset is **not included in this repository** due to file size. CIFAR-10 is downloaded automatically via `torchvision` on first run.

| Dataset | Download |
|---------|----------|
| Dogs vs. Cats (train / val / test) | [Google Drive — link to be added] |
| CIFAR-10 | Downloaded automatically via `torchvision` |

After downloading, place the Dogs vs. Cats dataset at:

```
data/
└── datasets/
    ├── train/
    │   ├── cat/
    │   └── dog/
    ├── val/
    │   ├── cat/
    │   └── dog/
    └── test/
        ├── 1.jpg
        ├── 2.jpg
        └── ...
```

Label mapping: `cat = 0`, `dog = 1`.

---

## Project Structure

```
.
├── models.py                   # CNN and ResNet-18 model definitions
├── train.py                    # Dogs vs. Cats training script
├── predict.py                  # Generate submission.csv from test set
├── submission.csv              # Test set predictions
│
├── utils/
│   ├── dataset.py              # Data loading and augmentation pipeline
│   ├── split_dataset.py        # Dataset splitting utility
│   └── dataset.md
│
├── tools/
│   ├── run_grid_search.py      # Grid search over hyperparameters
│   ├── run_experiments.py      # Batch experiment runner
│   ├── evaluate_val.py         # Validation set evaluation
│   └── visualize_errors.py     # Misclassified sample visualisation
│
├── scripts/
│   ├── run_grid_search_bs128.sh
│   └── run_grid_search_bs64.sh
│
├── cifar10/                    # CIFAR-10 extension (Part C)
│   ├── cifar10_dataset.py
│   ├── train_cifar10.py
│   ├── eval_cifar10.py
│   ├── eval_dogcat.py
│   ├── plot_cifar10_comparison.py
│   ├── run_cifar10.sh
│   ├── run_eval_cifar10.sh
│   ├── eval_dogcat.sh
│   └── README.md
│
├── outputs/                    # Experiment results (model weights excluded)
├── notebooks/
│   └── dataset_demo.ipynb
└── Description/
    ├── EE6483-Project2.pdf
    └── EE6483_project_briefing.pdf
```

---

## Installation

```bash
conda create -n ee6483 python=3.9
conda activate ee6483
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib scikit-learn tqdm pillow pandas
sudo apt-get update && sudo apt-get install -y tree
```

---

## Dogs vs. Cats

> For a complete step-by-step workflow including dataset verification, training, submission generation, and error visualisation, see [`README_workflow.md`](README_workflow.md).

### 1. Verify Dataset

```bash
tree -L 3 data/datasets
```

Quick sanity check:

```bash
python - <<'PY'
from utils.dataset import DatasetConfig, build_dataloaders
config = DatasetConfig(root_dir="./data/datasets", img_size=224, batch_size=8, num_workers=0, augmentation="none")
train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(config)
print("class_to_idx =", class_to_idx)
print("train batches =", len(train_loader))
print("val   batches =", len(val_loader))
PY
```

### 2. Training

Debug run (fast check):

```bash
python train.py \
    --data_root ./data/datasets \
    --model cnn \
    --epochs 2 \
    --batch_size 16 \
    --lr 0.001 \
    --augmentation light \
    --train_subset 200 \
    --val_subset 100 \
    --output_dir outputs \
    --run_name debug_run
```

Full training run:

```bash
python train.py \
    --data_root ./data/datasets \
    --model resnet \
    --pretrained \
    --unfreeze_layer4 \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.0001 \
    --optimizer adamw \
    --scheduler cosine \
    --augmentation strong \
    --output_dir outputs/grid_search_fixed_ep50_bs128 \
    --run_name resnet_aug-strong_lr-0.0001_bs-128_opt-adamw_sch-cosine_ep-50
```

Or run the full grid search via Slurm:

```bash
sbatch scripts/run_grid_search_bs128.sh
```

Training outputs are saved under `outputs/<run_name>/`:

```
acc_curve.png
loss_curve.png
model_best.pth
summary.json
```

### 3. Generate Submission

```bash
python predict.py \
    --data_root ./data/datasets \
    --model resnet \
    --checkpoint outputs/grid_search_fixed_ep50_bs128/resnet_aug-strong_lr-0.0001_bs-128_opt-adamw_sch-cosine_ep-50/model_best.pth \
    --pretrained \
    --unfreeze_layer4 \
    --output_csv submission.csv
```

Verify:

```bash
head submission.csv
wc -l submission.csv
```

### 4. Evaluate on Validation Set

```bash
PYTHONPATH=. python tools/evaluate_val.py \
    --data_root ./data/datasets \
    --model resnet \
    --checkpoint outputs/grid_search_fixed_ep50_bs128/resnet_aug-strong_lr-0.0001_bs-128_opt-adamw_sch-cosine_ep-50/model_best.pth \
    --pretrained \
    --unfreeze_layer4 \
    --output_json outputs/grid_search_fixed_ep50_bs128/resnet_aug-strong_lr-0.0001_bs-128_opt-adamw_sch-cosine_ep-50/val_metrics.json
```

### 5. Visualise Misclassified Samples

```bash
PYTHONPATH=. python tools/visualize_errors.py \
    --data_root ./data/datasets \
    --model resnet \
    --checkpoint outputs/grid_search_fixed_ep50_bs128/resnet_aug-strong_lr-0.0001_bs-128_opt-adamw_sch-cosine_ep-50/model_best.pth \
    --pretrained \
    --unfreeze_layer4 \
    --output_png outputs/case_study/val_errors.png
```

### Best Result

| Model | Augmentation | LR | Optimizer | Scheduler | Val Acc |
|-------|--------------|----|-----------|-----------|---------|
| ResNet-18 | strong | 1e-4 | AdamW | cosine | 99.12% |

---

## CIFAR-10 Extension

> For full documentation on training commands, experiment configurations, and evaluation, see [`cifar10/README.md`](cifar10/README.md).

**Summary of imbalance experiments** (bird, cat, deer, dog downsampled to 20%):

| Experiment | Strategy | Val Acc |
|------------|----------|---------|
| Exp 1 — Balanced Baseline | None (upper bound) | 89.32% |
| Exp 4 — Imbalanced Baseline | None (lower bound) | 83.20% |
| Exp 2 — Class Weights | Weighted CrossEntropyLoss | 83.71% |
| Exp 3 — Oversampling | WeightedRandomSampler | 83.98% |

---

## Team Contributions

| Member | Responsibilities |
|--------|-----------------|
| A | Model definition, training pipeline, hyperparameter tuning, model comparison |
| B | Data loading, preprocessing, augmentation, submission.csv, error visualisation |
| C | CIFAR-10 extension, class imbalance handling, code integration |
