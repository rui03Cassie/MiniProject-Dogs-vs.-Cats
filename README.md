# EE6483 Mini Project — Dogs vs. Cats & CIFAR-10

Binary and multi-class image classification project for EE6483 Artificial Intelligence and Data Mining (NTU).

---

## Repository Structure

This repository contains four branches:

| Branch | Description |
|--------|-------------|
| `main` | Merged codebase |
| `dev_A` | Model definition, training pipeline, prediction |
| `dev_B` | Data loading, preprocessing, augmentation, grid search |
| `dev_C` | CIFAR-10 extension, class imbalance handling, case study |

Each branch contains its own README with detailed usage instructions:
- `dev_A` — see `training.md`
- `dev_B` — see `README_workflow.md`
- `dev_C` — see `cifar10/README.md`

---

## Project Overview

```
.
├── models.py               # CNN and ResNet-18 and ResNet36 model definitions
├── train.py                # Dogs vs. Cats training script
├── predict.py              # Generate submission.csv from test set
├── submission.csv          # Final test set predictions
├── utils/                  # Data loading and preprocessing
├── tools/                  # Grid search, evaluation, visualisation
├── scripts/                # Slurm job scripts
├── cifar10/                # CIFAR-10 extension
└── outputs/                # Experiment results (model weights excluded)
```

---

## Dataset

The Dogs vs. Cats dataset is not included in this repository.

| Resource | Link |
|----------|------|
| Dogs vs. Cats dataset (train / val / test) | [https://drive.google.com/file/d/1q0r6yeHQMS17R3wz-s2FIbMR5DAGZK5v/view] |
| Pretrained model checkpoints and experiments outputs | [https://drive.google.com/drive/folders/1e4AjQ0f6xcoq4e-HvHvJTe3RTX8SWKeA?usp=sharing] |

CIFAR-10 is downloaded automatically via `torchvision` on first run.

After downloading, place the Dogs vs. Cats dataset at:

```
data/datasets/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
```

---

## Installation

```bash
conda create -n ee6483 python=3.9
conda activate ee6483
pip install torch torchvision numpy pandas matplotlib tqdm pillow scikit-learn
```

---

## Quick Start

### Dogs vs. Cats

**Train:**
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
    --output_dir outputs/run1 \
    --run_name resnet_run
```

**Predict:**
```bash
python predict.py \
    --data_root ./data/datasets \
    --model resnet \
    --checkpoint outputs/run1/resnet_run/model_best.pth \
    --pretrained \
    --unfreeze_layer4 \
    --output_csv submission.csv
```

For the full workflow including dataset verification, validation evaluation, and error visualisation, see `README_workflow.md` on `dev_B`.

### CIFAR-10 Extension

```bash
PYTHONPATH=. python cifar10/train_cifar10.py \
    --model cnn \
    --imbalance_method none \
    --epochs 40 \
    --batch_size 256 \
    --output_dir outputs/cifar10 \
    --run_name exp1_baseline
```

For all four imbalance experiments and evaluation, see `cifar10/README.md` on `dev_C`.

---

## Results

### Dogs vs. Cats (Validation Set)

Full grid search results: `outputs/grid_search_fixed_ep50_bs128/grid_results.csv`

**CNN — Top 3**

| Rank | Aug | LR | Batch | Opt | Sch | Val Acc |
|------|-----|----|-------|-----|-----|---------|
| 1 | light | 0.0005 | 32 | adam | plateau | 98.22% |
| 2 | light | 0.0005 | 32 | adam | cosine | 98.18% |
| 3 | light | 0.0005 | 32 | adam | cosine | 98.18% |

**ResNet-18 — Top 3**

| Rank | Aug | LR | Batch | Opt | Sch | Val Acc |
|------|-----|----|-------|-----|-----|---------|
| 1 | strong | 0.0001 | 32 | adamw | plateau | 99.22% |
| 2 | light | 0.0001 | 32 | adam | plateau | 99.18% |
| 3 | strong | 0.0001 | 128 | adam | cosine | 99.18% |

**ResNet-34 — Top 3**

| Rank | Aug | LR | Batch | Opt | Sch | Val Acc |
|------|-----|----|-------|-----|-----|---------|
| 1 | none | 0.0001 | 128 | adamw | plateau | 99.32% |
| 2 | strong | 0.0001 | 128 | adamw | plateau | 99.30% |
| 3 | strong | 0.0001 | 32 | adamw | cosine | 99.30% |

### CIFAR-10 Imbalance Study

| Experiment | Strategy | Val Acc |
|------------|----------|---------|
| Exp 1 — Balanced Baseline | None (upper bound) | 89.32% |
| Exp 4 — Imbalanced Baseline | None (lower bound) | 83.20% |
| Exp 2 — Class Weights | Weighted CrossEntropyLoss | 83.71% |
| Exp 3 — Oversampling | WeightedRandomSampler | 83.98% |

---

## Team

| Member | Branch | Responsibilities |
|--------|--------|-----------------|
| Wei Zishan | `dev_A` | Model definition, training pipeline, prediction |
| Mao Yuhui  | `dev_B` | Data loading, preprocessing, augmentation, grid search |
| Chen Lei | `dev_C` | CIFAR-10 extension, imbalance handling, case study |
