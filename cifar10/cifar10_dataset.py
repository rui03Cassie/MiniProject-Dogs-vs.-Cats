"""
cifar10_dataset.py — CIFAR-10 dataloader for EE6483 Mini Project (Part C Extension)

Mirrors the interface of utils/dataset.py so that train_cifar10.py can reuse
the same training loop from train.py with minimal changes.

CIFAR-10 classes (alphabetical order, matches torchvision):
    0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
    5: dog, 6: frog, 7: horse, 8: ship, 9: truck
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 添加根目录

import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


# ── CIFAR-10 statistics ────────────────────────────────────────────────────────
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = 10


# ── Config dataclass (mirrors DatasetConfig) ───────────────────────────────────
@dataclass
class CIFAR10Config:
    data_root: str = "./cifar10_data"   # torchvision downloads here
    img_size: int = 32                  # CIFAR-10 native size; set 224 for ResNet
    batch_size: int = 128
    num_workers: int = 4
    augmentation: str = "light"         # "none" | "light" | "strong"
    imbalance_method: str = "none"  # "none" | "class_weight" | "oversample" | "imbalanced_only"
    # If imbalance_method != "none", simulate imbalance by downsampling these classes:
    minority_classes: list = field(default_factory=lambda: [2, 3, 4, 5])  # bird/cat/deer/dog
    minority_ratio: float = 0.2         # keep only 20% of minority class samples
    seed: int = 42


# ── Transforms ────────────────────────────────────────────────────────────────
def build_cifar10_train_transform(img_size: int = 32, augmentation: str = "light"):
    norm = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    if augmentation == "none":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm,
        ])
    elif augmentation == "light":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            norm,
        ])
    elif augmentation == "strong":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            norm,
        ])
    else:
        raise ValueError(f"Unsupported augmentation: {augmentation}")


def build_cifar10_eval_transform(img_size: int = 32):
    norm = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])


# ── Imbalance simulation ───────────────────────────────────────────────────────
def make_imbalanced_subset(dataset, minority_classes: list, minority_ratio: float, seed: int):
    """
    Simulate class imbalance by keeping only `minority_ratio` of samples
    belonging to `minority_classes`. All other classes are kept in full.
    """
    rng = random.Random(seed)
    kept_indices = []
    for idx, (_, label) in enumerate(dataset):
        if label in minority_classes:
            if rng.random() < minority_ratio:
                kept_indices.append(idx)
        else:
            kept_indices.append(idx)
    print(f"[Imbalance] Original size: {len(dataset)} → Subset size: {len(kept_indices)}")
    return Subset(dataset, kept_indices)


def get_class_weights(dataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Returns a tensor of shape [NUM_CLASSES].
    """
    counts = torch.zeros(NUM_CLASSES)
    for _, label in dataset:
        counts[label] += 1
    counts = counts.clamp(min=1)          # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES   # normalise so mean ≈ 1
    print("[Class weights]", {CIFAR10_CLASSES[i]: f"{weights[i]:.3f}" for i in range(NUM_CLASSES)})
    return weights


def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that over-samples minority classes
    so each class appears roughly equally within one epoch.
    """
    counts = torch.zeros(NUM_CLASSES)
    labels = []
    for _, label in dataset:
        counts[label] += 1
        labels.append(label)
    counts = counts.clamp(min=1)
    class_weights = 1.0 / counts
    sample_weights = torch.tensor([class_weights[lbl] for lbl in labels])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print("[Oversampling] WeightedRandomSampler created.")
    return sampler


# ── Main builder ──────────────────────────────────────────────────────────────
def build_cifar10_dataloaders(config: CIFAR10Config):
    """
    Returns:
        train_loader, val_loader, class_weights (or None), class_to_idx

    - val_loader uses the official CIFAR-10 test split (10 000 images).
    - class_weights is a Tensor[10] when config.imbalance_method == "class_weight",
      otherwise None. Pass it to nn.CrossEntropyLoss(weight=class_weights).
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_transform = build_cifar10_train_transform(config.img_size, config.augmentation)
    eval_transform  = build_cifar10_eval_transform(config.img_size)

    train_dataset = datasets.CIFAR10(
        root=config.data_root, train=True,  download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=config.data_root, train=False, download=True, transform=eval_transform
    )

    # ── Simulate imbalance if requested ───────────────────────────────────────
    class_weights = None
    sampler = None

    if config.imbalance_method in ("class_weight", "oversample", "imbalanced_only"):
        train_dataset = make_imbalanced_subset(
            train_dataset,
            config.minority_classes,
            config.minority_ratio,
            config.seed,
        )

    if config.imbalance_method == "class_weight":
        class_weights = get_class_weights(train_dataset)

    elif config.imbalance_method == "oversample":
        sampler = get_weighted_sampler(train_dataset)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    generator = torch.Generator().manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),   # shuffle=False when using sampler
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    class_to_idx = {name: i for i, name in enumerate(CIFAR10_CLASSES)}
    return train_loader, val_loader, class_weights, class_to_idx


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    for method in ["none", "class_weight", "oversample"]:
        print(f"\n{'='*50}")
        print(f"Testing imbalance_method = '{method}'")
        cfg = CIFAR10Config(imbalance_method=method, batch_size=64, num_workers=0)
        train_loader, val_loader, class_weights, class_to_idx = build_cifar10_dataloaders(cfg)
        imgs, labels = next(iter(train_loader))
        print(f"  Train batch: images={tuple(imgs.shape)}, labels={tuple(labels.shape)}")
        print(f"  Val   size : {len(val_loader.dataset)}")
        if class_weights is not None:
            print(f"  Class weights: {class_weights.tolist()}")
        print(f"  class_to_idx: {class_to_idx}")