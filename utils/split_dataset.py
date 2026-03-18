#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_dataset.py

适用于原始目录结构：
dogs-vs-cats/
├── train/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   └── dog.1.jpg
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...

生成新结构：
dogs-vs-cats/
├── train/   # 原始，不动
├── test/    # 原始，不动
└── data/
    ├── train/
    │   ├── cat/
    │   └── dog/
    ├── val/
    │   ├── cat/
    │   └── dog/
    └── test/

默认使用复制，不破坏原始数据。
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASSES = {"cat", "dog"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_move_file(src: Path, dst: Path, use_copy: bool) -> None:
    if dst.exists():
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def parse_class_from_filename(filename: str) -> str:
    """
    例如:
      cat.11.jpg -> cat
      dog.5321.jpg -> dog
    """
    parts = filename.split(".")
    if len(parts) < 3:
        raise ValueError(f"Unexpected training filename format: {filename}")
    cls = parts[0].lower()
    if cls not in VALID_CLASSES:
        raise ValueError(f"Unknown class prefix in filename: {filename}")
    return cls


def get_flat_train_files(train_dir: Path) -> Dict[str, List[Path]]:
    """
    从扁平 train/ 中读取文件，并按 cat / dog 分组
    """
    grouped = {"cat": [], "dog": []}

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing original train folder: {train_dir}")

    all_files = [p for p in train_dir.iterdir() if is_image_file(p)]
    if len(all_files) == 0:
        raise RuntimeError(f"No training images found in: {train_dir}")

    for p in all_files:
        cls = parse_class_from_filename(p.name)
        grouped[cls].append(p)

    return grouped


def get_test_files(test_dir: Path) -> List[Path]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing original test folder: {test_dir}")

    files = sorted([p for p in test_dir.iterdir() if is_image_file(p)])
    if len(files) == 0:
        raise RuntimeError(f"No test images found in: {test_dir}")
    return files


def build_output_dirs(data_dir: Path) -> None:
    for split in ["train", "val"]:
        for cls in ["cat", "dog"]:
            ensure_dir(data_dir / split / cls)
    ensure_dir(data_dir / "test")


def count_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len([p for p in folder.iterdir() if p.is_file()])


def split_train_val(
    grouped_files: Dict[str, List[Path]],
    data_dir: Path,
    val_ratio: float,
    seed: int,
    use_copy: bool,
) -> None:
    random.seed(seed)

    for cls in ["cat", "dog"]:
        files = grouped_files[cls][:]
        random.shuffle(files)

        n_total = len(files)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        val_files = files[:n_val]
        train_files = files[n_val:]

        print(f"[{cls}] total={n_total}, train={n_train}, val={n_val}")

        for f in train_files:
            dst = data_dir / "train" / cls / f.name
            copy_or_move_file(f, dst, use_copy)

        for f in val_files:
            dst = data_dir / "val" / cls / f.name
            copy_or_move_file(f, dst, use_copy)


def prepare_test_set(test_files: List[Path], data_dir: Path, use_copy: bool) -> None:
    for f in test_files:
        dst = data_dir / "test" / f.name
        copy_or_move_file(f, dst, use_copy)

    print(f"[test] total={len(test_files)}")


def print_summary(data_dir: Path) -> None:
    print("\n=== Output Summary ===")
    for split in ["train", "val"]:
        for cls in ["cat", "dog"]:
            folder = data_dir / split / cls
            print(f"{split}/{cls}: {count_files(folder)}")
    print(f"test: {count_files(data_dir / 'test')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split flat Dogs vs Cats dataset into data/train, data/val, data/test"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the original dogs-vs-cats root directory",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for each class, default=0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed, default=42",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into data/ (recommended, default behavior)",
    )
    mode_group.add_argument(
        "--move",
        action="store_true",
        help="Move files into data/ instead of copying",
    )

    parser.add_argument(
        "--force_clean",
        action="store_true",
        help="Delete existing data/ folder before generating a new split",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    raw_train_dir = root_dir / "train"
    raw_test_dir = root_dir / "test"
    data_dir = root_dir / "data"

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be between 0 and 1")

    use_copy = True
    if args.move:
        use_copy = False

    grouped_files = get_flat_train_files(raw_train_dir)
    test_files = get_test_files(raw_test_dir)

    if data_dir.exists() and args.force_clean:
        print(f"Removing existing folder: {data_dir}")
        shutil.rmtree(data_dir)

    build_output_dirs(data_dir)

    print(f"Root dir: {root_dir}")
    print(f"Output dir: {data_dir}")
    print(f"Mode: {'copy' if use_copy else 'move'}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Seed: {args.seed}\n")

    split_train_val(
        grouped_files=grouped_files,
        data_dir=data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_copy=use_copy,
    )

    prepare_test_set(
        test_files=test_files,
        data_dir=data_dir,
        use_copy=use_copy,
    )

    print_summary(data_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()