from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    "cnn": {
        "extra_args": [],
        "default_lr": 1e-3,
        "default_batch_size": 32,
    },
    "resnet": {
        "extra_args": ["--pretrained", "--unfreeze_layer4"],
        "default_lr": 1e-4,
        "default_batch_size": 32,
    },
}

AUGMENTATIONS = ["none", "light", "strong"]


def run_command(cmd: list[str]) -> int:
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 100)
    result = subprocess.run(cmd)
    return result.returncode


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/datasets")
    parser.add_argument("--output_dir", type=str, default="outputs/experiments")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--val_subset", type=int, default=None)

    parser.add_argument("--models", nargs="*", default=["cnn", "resnet"])
    parser.add_argument("--augmentations", nargs="*", default=AUGMENTATIONS)

    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "experiment_results.csv"
    rows = []

    for model_name, aug in itertools.product(args.models, args.augmentations):
        if model_name not in EXPERIMENTS:
            print(f"Skip unsupported model: {model_name}")
            continue

        cfg = EXPERIMENTS[model_name]
        run_name = f"{model_name}_aug-{aug}"
        run_dir = output_dir / run_name
        summary_path = run_dir / "summary.json"

        if summary_path.exists() and not args.force:
            print(f"Skip existing run: {run_name}")
            summary = load_summary(summary_path)
        else:
            cmd = [
                sys.executable, "train.py",
                "--data_root", args.data_root,
                "--model", model_name,
                "--epochs", str(args.epochs),
                "--batch_size", str(cfg["default_batch_size"]),
                "--lr", str(cfg["default_lr"]),
                "--augmentation", aug,
                "--num_workers", str(args.num_workers),
                "--seed", str(args.seed),
                "--output_dir", str(output_dir),
                "--run_name", run_name,
            ]

            if args.train_subset is not None:
                cmd += ["--train_subset", str(args.train_subset)]
            if args.val_subset is not None:
                cmd += ["--val_subset", str(args.val_subset)]

            cmd += cfg["extra_args"]

            code = run_command(cmd)
            if code != 0:
                print(f"Run failed: {run_name}")
                rows.append({
                    "run_name": run_name,
                    "model": model_name,
                    "augmentation": aug,
                    "status": "failed",
                })
                continue

            summary = load_summary(summary_path)

        row = {
            "run_name": run_name,
            "model": model_name,
            "augmentation": aug,
            "status": "ok",
            "best_epoch": summary.get("best_epoch"),
            "best_val_acc": summary.get("best_val_acc"),
            "lastEpoch_val_acc": summary.get("lastEpoch_val_acc"),
            "epochs_nums": summary.get("epochs_nums"),
            "total_time_sec": summary.get("total_time_sec"),
            "device": summary.get("device"),
            "lr": summary.get("config", {}).get("lr"),
            "batch_size": summary.get("config", {}).get("batch_size"),
        }
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            x.get("status") != "ok",
            -(x.get("best_val_acc") or -1),
        )
    )

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_name", "model", "augmentation", "status",
            "best_epoch", "best_val_acc", "lastEpoch_val_acc",
            "epochs_nums", "total_time_sec", "device", "lr", "batch_size"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    print("\nFinished all experiments.")
    print(f"Saved summary table to: {results_csv}")


if __name__ == "__main__":
    main()
