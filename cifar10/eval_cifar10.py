"""
eval_cifar10.py — Per-class evaluation for all 4 CIFAR-10 experiments.

Outputs per experiment (saved under outputs/cifar10/<run_name>/):
  - val_metrics.json   : accuracy, per-class acc, confusion matrix, classification report
  - confusion_matrix.png
  - sample_grid.png    : 4 correct + 4 misclassified samples with confidence scores

Usage:
    python eval_cifar10.py
    python eval_cifar10.py --output_dir outputs/cifar10 --data_root ./cifar10_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent if _HERE.parent.name == "cifar10" else _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets

from models import build_model
from cifar10_dataset import (
    CIFAR10Config,
    CIFAR10_CLASSES,
    NUM_CLASSES,
    build_cifar10_eval_transform,
)

# ── Experiment definitions ────────────────────────────────────────────────────
EXPERIMENTS = [
    dict(run_name="exp1_baseline",            label="Exp1 Balanced Baseline"),
    dict(run_name="exp2_class_weight",         label="Exp2 Class Weights"),
    dict(run_name="exp3_oversample",           label="Exp3 Oversampling"),
    dict(run_name="exp4_imbalanced_baseline",  label="Exp4 Imbalanced Baseline"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model("cnn", num_classes=NUM_CLASSES, dropout=0.3)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def run_inference(model, data_root: str, img_size: int, batch_size: int,
                  device: torch.device):
    """Run model over CIFAR-10 test split, return (all_preds, all_labels, all_probs, all_images)."""
    transform = build_cifar10_eval_transform(img_size)
    dataset = datasets.CIFAR10(root=data_root, train=False, download=True,
                               transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    all_preds, all_labels, all_probs = [], [], []
    # Also keep raw PIL images for sample visualisation (first pass only)
    raw_dataset = datasets.CIFAR10(root=data_root, train=False, download=False)

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs = F.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    return all_preds, all_labels, all_probs, raw_dataset


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_sample_grid(all_preds, all_labels, all_probs, raw_dataset,
                     save_path: Path, title: str, n_each: int = 4):
    """
    Show n_each correctly classified + n_each misclassified samples.
    Picks highest-confidence correct and lowest-confidence wrong samples
    (the interesting edge cases).
    """
    correct_idx = np.where(all_preds == all_labels)[0]
    wrong_idx   = np.where(all_preds != all_labels)[0]

    # Sort: highest conf correct, lowest conf wrong (most confusing)
    correct_idx = sorted(correct_idx, key=lambda i: -all_probs[i][all_preds[i]])
    wrong_idx   = sorted(wrong_idx,   key=lambda i:  all_probs[i][all_preds[i]])

    selected = (
        [(i, True)  for i in correct_idx[:n_each]] +
        [(i, False) for i in wrong_idx[:n_each]]
    )

    ncols = n_each
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for row, correct in enumerate([True, False]):
        subset = [(i, c) for i, c in selected if c == correct]
        row_title = "Correctly Classified" if correct else "Misclassified"
        for col, (idx, _) in enumerate(subset):
            ax = axes[row][col]
            img, _ = raw_dataset[idx]          # PIL image, no transform
            ax.imshow(img)
            true_cls = CIFAR10_CLASSES[all_labels[idx]]
            pred_cls = CIFAR10_CLASSES[all_preds[idx]]
            conf     = all_probs[idx][all_preds[idx]]
            color    = "green" if correct else "red"
            ax.set_title(
                f"True: {true_cls}\nPred: {pred_cls}\nConf: {conf:.2f}",
                fontsize=8, color=color,
            )
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_title, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Per-class accuracy helper ─────────────────────────────────────────────────
def per_class_accuracy(all_preds, all_labels):
    accs = {}
    for c, name in enumerate(CIFAR10_CLASSES):
        mask = all_labels == c
        accs[name] = float((all_preds[mask] == c).mean()) if mask.sum() > 0 else 0.0
    return accs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/cifar10")
    parser.add_argument("--data_root",  default="./cifar10_data")
    parser.add_argument("--img_size",   type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_root = Path(args.output_dir)

    for exp in EXPERIMENTS:
        run_name = exp["run_name"]
        label    = exp["label"]
        run_dir  = output_root / run_name
        ckpt     = run_dir / "model_best.pth"

        print(f"\n{'='*55}")
        print(f"Evaluating: {label}  ({run_name})")

        if not ckpt.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt}")
            continue

        # ── Inference ─────────────────────────────────────────────────────
        model = load_model(ckpt, device)
        all_preds, all_labels, all_probs, raw_dataset = run_inference(
            model, args.data_root, args.img_size, args.batch_size, device
        )

        # ── Metrics ───────────────────────────────────────────────────────
        acc    = float((all_preds == all_labels).mean())
        cm     = confusion_matrix(all_labels, all_preds).tolist()
        report = classification_report(
            all_labels, all_preds,
            target_names=CIFAR10_CLASSES,
            output_dict=True,
        )
        per_cls_acc = per_class_accuracy(all_preds, all_labels)

        metrics = {
            "experiment":       run_name,
            "overall_accuracy": acc,
            "per_class_accuracy": per_cls_acc,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        metrics_path = run_dir / "val_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Overall accuracy: {acc:.4f}")
        print(f"  Per-class accuracy:")
        for cls, a in per_cls_acc.items():
            tag = " ← minority" if cls in ("bird","cat","deer","dog") else ""
            print(f"    {cls:12s}: {a:.4f}{tag}")
        print(f"  Saved: {metrics_path}")

        # ── Plots ──────────────────────────────────────────────────────────
        plot_confusion_matrix(
            np.array(cm),
            run_dir / "confusion_matrix.png",
            title=f"Confusion Matrix — {label}",
        )
        plot_sample_grid(
            all_preds, all_labels, all_probs, raw_dataset,
            run_dir / "sample_grid.png",
            title=f"Sample Cases — {label}",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()