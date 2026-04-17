# CIFAR-10 Extension

This module extends the Dogs vs. Cats binary classifier to the 10-class CIFAR-10 benchmark, and investigates two strategies for handling class imbalance. It corresponds to sections g) and h) of the EE6483 Mini Project report.

---

## File Overview

| File | Description |
|------|-------------|
| `cifar10_dataset.py` | Data loading and preprocessing for CIFAR-10. Defines `CIFAR10Config`, build functions for train/val transforms, imbalance simulation (`make_imbalanced_subset`), class weight computation, and `WeightedRandomSampler` construction. |
| `train_cifar10.py` | Training script. Supports CNN and ResNet backbones, all four imbalance methods (`none`, `class_weight`, `oversample`, `imbalanced_only`), cosine/plateau/step schedulers, AMP, and early stopping. Saves `model_best.pth`, `summary.json`, and training curves. |
| `eval_cifar10.py` | Inference script. Loads `model_best.pth` for each experiment, runs evaluation on the CIFAR-10 test split, and outputs `val_metrics.json` (overall accuracy, per-class accuracy, confusion matrix), `confusion_matrix.png`, and `sample_grid.png`. |
| `eval_dogcat.py` | Case study inference script for the Dogs vs. Cats model. Finds high-confidence correct samples and misclassified samples from the validation set, outputs annotated images and `case_study.json`. |
| `plot_cifar10_comparison.py` | Generates the cross-experiment validation accuracy comparison plot (`comparison_val_acc.png`). |
| `run_cifar10.sh` | Slurm batch script for running all four CIFAR-10 training experiments sequentially. |
| `run_eval_cifar10.sh` | Slurm batch script for running `eval_cifar10.py` across all four experiments. |
| `eval_dogcat.sh` | Slurm batch script for running `eval_dogcat.py`. |

---

## How to Run

All scripts should be run from the **project root directory**, not from inside `cifar10/`.

### Training

Run a single experiment:

```bash
PYTHONPATH=. python cifar10/train_cifar10.py \
    --model cnn \
    --imbalance_method none \
    --epochs 40 \
    --batch_size 256 \
    --lr 0.001 \
    --augmentation light \
    --output_dir outputs/cifar10 \
    --run_name exp1_baseline
```

The four experimental configurations are:

| Experiment | `--imbalance_method` | `--run_name` |
|------------|----------------------|--------------|
| Exp 1 — Balanced Baseline | `none` | `exp1_baseline` |
| Exp 2 — Class Weights | `class_weight` | `exp2_class_weight` |
| Exp 3 — Oversampling | `oversample` | `exp3_oversample` |
| Exp 4 — Imbalanced Baseline | `imbalanced_only` | `exp4_imbalanced_baseline` |

To run all four via Slurm:

```bash
sbatch cifar10/run_cifar10.sh
```

### Evaluation

After training, run inference to generate per-class metrics and sample visualisations:

```bash
PYTHONPATH=. python cifar10/eval_cifar10.py \
    --output_dir outputs/cifar10 \
    --data_root ./cifar10_data \
    --img_size 32 \
    --batch_size 256
```

Or via Slurm:

```bash
sbatch cifar10/run_eval_cifar10.sh
```

Each experiment's output directory will contain:

```
outputs/cifar10/<run_name>/
    model_best.pth          # Best checkpoint (excluded from Git)
    summary.json            # Training summary
    val_metrics.json        # Per-class accuracy, confusion matrix, F1
    acc_curve.png
    loss_curve.png
    confusion_matrix.png
    sample_grid.png
```

---

## Experiment Results

CIFAR-10 images are kept at native 32×32 resolution. The training set is artificially imbalanced by downsampling four animal classes (bird, cat, deer, dog) to 20% of their original size (~1,000 samples each), while the remaining six classes retain all 5,000 samples. This gives an imbalance ratio of approximately 5:1.

| Experiment | Training Size | Strategy | Best Val Acc | Best / Total Epochs |
|------------|--------------|----------|--------------|---------------------|
| Exp 1 — Balanced Baseline | 50,000 | None (upper bound) | 89.32% | 40 / 40 |
| Exp 4 — Imbalanced Baseline | ~34,067 | None (lower bound) | 83.20% | 29 / 36 |
| Exp 2 — Class Weights | ~34,067 | Weighted CrossEntropyLoss | 83.71% | 40 / 40 |
| Exp 3 — Oversampling | ~34,067 | WeightedRandomSampler | 83.98% | 26 / 33 |

Per-class accuracy for the four minority classes:

| Class | Exp 1 | Exp 4 | Exp 2 | Exp 3 |
|-------|-------|-------|-------|-------|
| bird  | 86.8% | 62.0% | 72.8% | 72.1% |
| cat   | 77.1% | 56.9% | 59.1% | 61.5% |
| deer  | 88.2% | 84.6% | 78.2% | 78.9% |
| dog   | 83.7% | 74.6% | 73.5% | 74.9% |

Both correction methods recover a portion of the performance drop caused by imbalance. Oversampling marginally outperforms class-weighted loss (+0.27 pp) and converges 14 epochs earlier.