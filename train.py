import argparse
import random
import numpy as np
import torch
import os
from pathlib import Path
from utils.dataset import DatasetConfig, build_dataloaders
import torch.nn as nn
from models import build_model
import copy
import time
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, mode="max", min_delta=0.001):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0
        self.min_delta = min_delta

    def step(self, value):
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            updated = value > self.best + self.min_delta
        else:
            updated = value < self.best - self.min_delta

        if updated:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience

def train_one_epoch(model, loader, loss_fn, optimizer, device, use_amp=False):
    model.train()
    loss_meter = Average()
    acc_meter = Average()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    process = tqdm(loader, desc="Train", leave=False, mininterval=0.5)
    for images, labels in process:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            loss = loss_fn(outputs, labels)  # forward

        scaler.scale(loss).backward()  # backward
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.size(0)
        acc = compute_accuracy(outputs.detach(), labels)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        process.set_postfix(loss=f"{loss_meter.avg():.4f}", acc=f"{acc_meter.avg():.4f}")  # 显示进度
    return loss_meter.avg(), acc_meter.avg()


class Average:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, batch_size=1):
        self.total += value * batch_size
        self.count += batch_size

    def avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


def compute_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == targets).float().mean().item()
    return accuracy


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_meter = Average()
    acc_meter = Average()

    pbar = tqdm(loader, desc="Val", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)  # forward
        batch_size = labels.size(0)
        acc = compute_accuracy(outputs, labels)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        pbar.set_postfix(loss=f"{loss_meter.avg():.4f}", acc=f"{acc_meter.avg():.4f}")  # 显示进度
    return loss_meter.avg(), acc_meter.avg()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augmentation", type=str, default="light")
    parser.add_argument("--normalize_mode", type=str, default="imagenet")
    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--val_subset", type=int, default=None)
    # model
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--unfreeze_layer4", action="store_true")
    # training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", type=str, default="plateau")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    #print("args=", args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 训练设备
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("Device=", device)

    if args.run_name is None:
        run_name = f"{args.model}_lr{args.lr}_bs{args.batch_size}_aug{args.augmentation}"
    else:
        run_name = args.run_name
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = DatasetConfig(
        root_dir=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        normalize_mode=args.normalize_mode,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(config)
    print("class_to_idx=", class_to_idx)
    print("train_batches=", len(train_loader), "val_batches=", len(val_loader))

    model = build_model(args.model, 2, args.dropout, args.pretrained, args.train_backbone, args.unfreeze_layer4).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    params = [param for param in model.parameters() if param.requires_grad]
    # 优化器
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    # 调度器
    if args.scheduler.lower() == "none":
        scheduler = None
    elif args.scheduler.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)
    elif args.scheduler.lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2,)
    elif args.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    # print("history=", history)

    best_val_acc = -1
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stopper = EarlyStopping(patience=args.early_stop_patience, mode="min", min_delta=args.early_stop_min_delta)
    start_time = time.time() # 记录时间
    # 开始训练
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.use_amp)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

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

        print(f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_dir / "model_best.pth")
            print(f"Saved best model to: {save_dir / 'model_best.pth'}")

        if stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    epochs = list(range(1, len(history["train_loss"]) + 1))

    tag = f"lr-{args.lr}_aug-{args.augmentation}"
    # draw
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"loss_curve_{tag}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="train_acc")
    plt.plot(epochs, history["val_acc"], marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"acc_curve_{tag}.png", dpi=200)
    plt.close()

    if len(history["val_acc"]) > 0:
        lastEpoch_val_acc = history["val_acc"][-1]
    else:
        lastEpoch_val_acc = None

    summary = {
        "best_epoch": best_epoch, 
        "best_val_acc": best_val_acc, 
        "lastEpoch_val_acc": lastEpoch_val_acc,
        "epochs_nums": len(history["train_loss"]), 
        "total_time_sec": total_time, 
        "device": str(device), 
        "class_to_idx": class_to_idx, 
        "config": vars(args),
        }

    summary_path = save_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining Done")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()