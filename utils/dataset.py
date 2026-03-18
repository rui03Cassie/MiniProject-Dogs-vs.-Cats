"""
dataset.py for EE6483 Mini Project (Dogs vs Cats)

This file covers:
1) Load the pre-split dataset from data/train, data/val, data/test
2) Different preprocessing / augmentation functions for experiments
3) Build PyTorch datasets and dataloaders

Note:
- Dataset splitting should be done separately by split_dataset.py
- This file assumes the dataset has already been organized as:

data/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Avoid crashing on slightly corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetConfig:
    root_dir: str
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    augmentation: str = "light"      # "none", "light", "strong"
    normalize_mode: str = "imagenet" # "imagenet", "none"
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    shuffle_train: bool = True
    pin_memory: bool = True
    seed: int = 42


class TestImageDataset(Dataset):
    """
    Custom dataset for test images.

    Expected structure:
        root_dir/
            test/
                1.jpg
                2.jpg
                ...

    Returns:
        image_tensor, image_id (int), file_name (str)
    """
    def __init__(self, test_dir: str | Path, transform: Optional[Callable] = None) -> None:
        self.test_dir = Path(test_dir)
        self.transform = transform

        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

        self.image_paths = sorted(
            [p for p in self.test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS],
            key=lambda p: self._extract_image_id(p.name)
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in test directory: {self.test_dir}")

    @staticmethod
    def _extract_image_id(file_name: str) -> int:
        stem = Path(file_name).stem
        try:
            return int(stem)
        except ValueError:
            digits = "".join(ch for ch in stem if ch.isdigit())
            if digits:
                return int(digits)
            raise ValueError(
                f"Could not parse image id from test filename: {file_name}. "
                "Expected names such as '1.jpg', '2.jpg', etc."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_id = self._extract_image_id(image_path.name)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id, image_path.name


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def get_normalization(normalize_mode: str = "imagenet"):
    if normalize_mode == "imagenet":
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    if normalize_mode == "none":
        return transforms.Lambda(lambda x: x)
    raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")


def build_train_transform(
    img_size: int = 224,
    augmentation: str = "light",
    normalize_mode: str = "imagenet",
):
    """
    Build training transforms.

    augmentation options:
    - "none":  Resize -> ToTensor -> Normalize
    - "light": Resize -> HorizontalFlip -> Rotation -> ToTensor -> Normalize
    - "strong": RandomResizedCrop -> HorizontalFlip -> Rotation
                -> ColorJitter -> ToTensor -> Normalize
    """
    norm = get_normalization(normalize_mode)

    if augmentation == "none":
        tfms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm,
        ]
    elif augmentation == "light":
        tfms = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            norm,
        ]
    elif augmentation == "strong":
        tfms = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.ToTensor(),
            norm,
        ]
    else:
        raise ValueError(
            f"Unsupported augmentation profile: {augmentation}. "
            "Choose from: 'none', 'light', 'strong'."
        )

    return transforms.Compose(tfms)


def build_eval_transform(
    img_size: int = 224,
    normalize_mode: str = "imagenet",
):
    """
    Build validation / test transforms.
    """
    norm = get_normalization(normalize_mode)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])


def _check_expected_structure(root_dir: str | Path) -> Dict[str, Path]:
    """
    Check whether the dataset has already been split into:
        root_dir/train/cat
        root_dir/train/dog
        root_dir/val/cat
        root_dir/val/dog
        root_dir/test
    """
    root = Path(root_dir)
    paths = {
        "train": root / "train",
        "val": root / "val",
        "test": root / "test",
    }

    for _, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected folder not found: {path}")

    for split in ["train", "val"]:
        for cls in ["cat", "dog"]:
            cls_path = paths[split] / cls
            if not cls_path.exists():
                raise FileNotFoundError(f"Expected class folder not found: {cls_path}")

    return paths


def _maybe_make_subset(dataset: Dataset, subset_size: Optional[int], seed: int) -> Dataset:
    if subset_size is None:
        return dataset
    if subset_size <= 0:
        raise ValueError("subset_size must be a positive integer or None.")

    subset_size = min(subset_size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_datasets(config: DatasetConfig):
    """
    Returns:
        train_dataset, val_dataset, test_dataset, class_to_idx

    Important:
    ImageFolder assigns labels alphabetically:
        {'cat': 0, 'dog': 1}
    This matches the coursework submission requirement:
        0 = cat, 1 = dog
    """
    seed_everything(config.seed)
    paths = _check_expected_structure(config.root_dir)

    train_transform = build_train_transform(
        img_size=config.img_size,
        augmentation=config.augmentation,
        normalize_mode=config.normalize_mode,
    )
    eval_transform = build_eval_transform(
        img_size=config.img_size,
        normalize_mode=config.normalize_mode,
    )

    train_dataset = datasets.ImageFolder(
        root=paths["train"],
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        root=paths["val"],
        transform=eval_transform,
    )
    test_dataset = TestImageDataset(
        test_dir=paths["test"],
        transform=eval_transform,
    )

    expected_mapping = {"cat": 0, "dog": 1}
    if train_dataset.class_to_idx != expected_mapping:
        raise RuntimeError(
            f"Unexpected class mapping: {train_dataset.class_to_idx}. "
            f"Expected: {expected_mapping}"
        )

    train_dataset = _maybe_make_subset(train_dataset, config.train_subset, config.seed)
    val_dataset = _maybe_make_subset(val_dataset, config.val_subset, config.seed)

    return train_dataset, val_dataset, test_dataset, expected_mapping


def build_dataloaders(config: DatasetConfig):
    """
    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """
    train_dataset, val_dataset, test_dataset, class_to_idx = build_datasets(config)

    generator = torch.Generator().manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader, test_loader, class_to_idx


def describe_dataset(config: DatasetConfig) -> Dict[str, object]:
    train_dataset, val_dataset, test_dataset, class_to_idx = build_datasets(config)

    info = {
        "root_dir": config.root_dir,
        "img_size": config.img_size,
        "augmentation": config.augmentation,
        "normalize_mode": config.normalize_mode,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "class_to_idx": class_to_idx,
    }
    return info


if __name__ == "__main__":
    # Example usage:
    # After running split_dataset.py, use the generated data/ folder.
    #
    # Expected structure:
    # dogs-vs-cats/
    #   data/
    #     train/cat
    #     train/dog
    #     val/cat
    #     val/dog
    #     test/

    config = DatasetConfig(
        root_dir="./dogs-vs-cats/data",
        img_size=224,
        batch_size=16,
        num_workers=2,
        augmentation="light",   # "none", "light", "strong"
        normalize_mode="imagenet",
        train_subset=None,
        val_subset=None,
        seed=42,
    )

    info = describe_dataset(config)
    print("Dataset summary:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(config)

    train_images, train_labels = next(iter(train_loader))
    print("\nOne training batch:")
    print("  images shape:", tuple(train_images.shape))
    print("  labels shape:", tuple(train_labels.shape))
    print("  label sample:", train_labels[:8].tolist())

    test_images, test_ids, test_names = next(iter(test_loader))
    print("\nOne test batch:")
    print("  images shape:", tuple(test_images.shape))
    print("  ids sample:", test_ids[:8].tolist())
    print("  names sample:", list(test_names[:8]))