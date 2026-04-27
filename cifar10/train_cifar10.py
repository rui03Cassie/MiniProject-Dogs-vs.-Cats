"""
train_cifar10.py — Training script for CIFAR-10 (Part C Extension)

Reuses models.py (CNN / ResNet) and the training loop logic from train.py.
Key differences vs train.py:
  - num_classes = 10
  - Uses cifar10_dataset.py for data loading
  - Supports --imbalance_method (none | class_weight | oversample)
  - CrossEntropyLoss accepts optional class weights

Usage examples:

  # Baseline (no imbalance handling)
  python train_cifar10.py --model cnn --epochs 30 --output_dir outputs_cifar10

  # With class weights
  python train_cifar10.py --model resnet --pretrained --imbalance_method class_weight

  # With oversampling
  python train_cifar10.py --model cnn --imbalance_method oversample
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 添加根目录

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import build_model
from cifar10_dataset import CIFAR10Config, NUM_CLASSES, build_cifar10_dataloaders


# ── Helpers (identical to train.py) ──────────────────────────────────────────
class Average:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += value * n
        self.count += n

    def avg(self):
        return self.total / self.count if self.count else 0.0


def compute_accuracy(logits, targets):
    return (torch.argmax(logits, dim=1) == targets).float().mean().item()


class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# ── Train / eval loops ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, loss_fn, optimizer, device, use_amp=False):
    model.train()
    loss_m, acc_m = Average(), Average()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    pbar = tqdm(loader, desc="Train", leave=False, mininterval=0.5)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = labels.size(0)
        loss_m.update(loss.item(), n)
        acc_m.update(compute_accuracy(outputs.detach(), labels), n)
        pbar.set_postfix(loss=f"{loss_m.avg():.4f}", acc=f"{acc_m.avg():.4f}")
    return loss_m.avg(), acc_m.avg()


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_m, acc_m = Average(), Average()
    pbar = tqdm(loader, desc="Val", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        n = labels.size(0)
        loss_m.update(loss.item(), n)
        acc_m.update(compute_accuracy(outputs, labels), n)
        pbar.set_postfix(loss=f"{loss_m.avg():.4f}", acc=f"{acc_m.avg():.4f}")
    return loss_m.avg(), acc_m.avg()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train CNN/ResNet on CIFAR-10")
    # Data
    parser.add_argument("--data_root",        type=str,   default="./cifar10_data")
    parser.add_argument("--img_size",         type=int,   default=32)
    parser.add_argument("--batch_size",       type=int,   default=128)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--augmentation",     type=str,   default="light",
                        choices=["none", "light", "strong"])
    parser.add_argument("--imbalance_method", type=str,   default="none",
                        choices=["none", "class_weight", "oversample", "imbalanced_only"],
                        help="How to handle simulated class imbalance")
    parser.add_argument("--minority_ratio",   type=float, default=0.2,
                        help="Fraction of minority-class samples to keep (imbalance simulation)")
    # Model
    parser.add_argument("--model",            type=str,   default="cnn",
                        choices=["cnn", "resnet"])
    parser.add_argument("--dropout",          type=float, default=0.3)
    parser.add_argument("--pretrained",       action="store_true")
    parser.add_argument("--train_backbone",   action="store_true")
    parser.add_argument("--unfreeze_layer4",  action="store_true")
    # Training
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--optimizer",        type=str,   default="adam",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler",        type=str,   default="cosine",
                        choices=["none", "step", "plateau", "cosine"])
    parser.add_argument("--early_stop_patience", type=int, default=7)
    parser.add_argument("--use_amp",          action="store_true")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output_dir",       type=str,   default="outputs_cifar10")
    parser.add_argument("--run_name",         type=str,   default=None)
    args = parser.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("Device:", device)

    # ── Output dir ────────────────────────────────────────────────────────────
    run_name = args.run_name or (
        f"cifar10_{args.model}_lr{args.lr}_bs{args.batch_size}"
        f"_aug{args.augmentation}_imb{args.imbalance_method}"
    )
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    cfg = CIFAR10Config(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        imbalance_method=args.imbalance_method,
        minority_ratio=args.minority_ratio,
        seed=args.seed,
    )
    train_loader, val_loader, class_weights, class_to_idx = build_cifar10_dataloaders(cfg)
    print("class_to_idx:", class_to_idx)
    print(f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # num_classes=10 (key difference from Dogs-vs-Cats which used 2)
    model = build_model(
        args.model, NUM_CLASSES, args.dropout,
        args.pretrained, args.train_backbone, args.unfreeze_layer4,
    ).to(device)

    # ── Loss (with optional class weights) ───────────────────────────────────
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using weighted CrossEntropyLoss.")
    else:
        loss_fn = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # ── Scheduler ────────────────────────────────────────────────────────────
    if args.scheduler == "none":
        scheduler = None
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    else:  # cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc, best_epoch = -1, -1
    best_state = copy.deepcopy(model.state_dict())
    stopper = EarlyStopping(patience=args.early_stop_patience, mode="max")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.use_amp)
        val_loss,   val_acc   = evaluate(model, val_loader, loss_fn, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        print(f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_dir / "model_best.pth")
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

        if stopper.step(val_acc):
            print(f"Early stopping at epoch {epoch}.")
            break

    total_time = time.time() - start_time

    # ── Plots ─────────────────────────────────────────────────────────────────
    epochs_range = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs_range, history["val_loss"],   marker="o", label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("CIFAR-10 Training and Validation Loss")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.3); plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_acc"], marker="o", label="train_acc")
    plt.plot(epochs_range, history["val_acc"],   marker="o", label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Training and Validation Accuracy")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.3); plt.tight_layout()
    plt.savefig(save_dir / "acc_curve.png", dpi=200); plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "task": "CIFAR-10",
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "lastEpoch_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "epochs_run": len(history["train_loss"]),
        "total_time_sec": total_time,
        "device": str(device),
        "class_to_idx": class_to_idx,
        "config": vars(args),
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Training Done ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()