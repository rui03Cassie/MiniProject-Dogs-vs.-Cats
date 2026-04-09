from __future__ import annotations
import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


MODEL_CONFIGS = {
    "cnn": {
        "extra_args": [],
    },
    "resnet": {
        "extra_args": ["--pretrained", "--unfreeze_layer4"],
    },
}


def run_command(cmd: list[str]) -> int:
    print("\n" + "=" * 120)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 120)
    result = subprocess.run(cmd)
    return result.returncode


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/datasets")
    parser.add_argument("--output_dir", type=str, default="outputs/grid_search")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--models", nargs="*", default=["cnn", "resnet"])
    parser.add_argument("--augmentations", nargs="*", default=["none", "light", "strong"])
    parser.add_argument("--lrs", nargs="*", type=float, default=[1e-3, 5e-4, 1e-4])
    parser.add_argument("--batch_sizes", nargs="*", type=int, default=[32])
    parser.add_argument("--optimizers", nargs="*", default=["adam", "adamw"])
    parser.add_argument("--schedulers", nargs="*", default=["plateau"])
    parser.add_argument("--epochs_list", nargs="*", type=int, default=[10, 15])

    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--val_subset", type=int, default=None)

    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "grid_results.csv"
    rows = []

    grid = itertools.product(
        args.models,
        args.augmentations,
        args.lrs,
        args.batch_sizes,
        args.optimizers,
        args.schedulers,
        args.epochs_list,
    )

    for model, aug, lr, batch_size, optimizer, scheduler, epochs in grid:
        if model not in MODEL_CONFIGS:
            print(f"Skip unsupported model: {model}")
            continue

        run_name = (
            f"{model}"
            f"_aug-{aug}"
            f"_lr-{lr}"
            f"_bs-{batch_size}"
            f"_opt-{optimizer}"
            f"_sch-{scheduler}"
            f"_ep-{epochs}"
        )

        run_dir = output_dir / run_name
        summary_path = run_dir / "summary.json"

        if summary_path.exists() and not args.force:
            print(f"Skip existing run: {run_name}")
            summary = load_summary(summary_path)
        else:
            cmd = [
                sys.executable, "train.py",
                "--data_root", args.data_root,
                "--model", model,
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--augmentation", aug,
                "--optimizer", optimizer,
                "--scheduler", scheduler,
                "--num_workers", str(args.num_workers),
                "--seed", str(args.seed),
                "--output_dir", str(output_dir),
                "--run_name", run_name,
            ]

            if args.train_subset is not None:
                cmd += ["--train_subset", str(args.train_subset)]
            if args.val_subset is not None:
                cmd += ["--val_subset", str(args.val_subset)]

            cmd += MODEL_CONFIGS[model]["extra_args"]

            code = run_command(cmd)
            if code != 0:
                rows.append({
                    "run_name": run_name,
                    "model": model,
                    "augmentation": aug,
                    "lr": lr,
                    "batch_size": batch_size,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "epochs": epochs,
                    "status": "failed",
                })
                continue

            summary = load_summary(summary_path)

        row = {
            "run_name": run_name,
            "model": model,
            "augmentation": aug,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epochs": epochs,
            "status": "ok",
            "best_epoch": summary.get("best_epoch"),
            "best_val_acc": summary.get("best_val_acc"),
            "lastEpoch_val_acc": summary.get("lastEpoch_val_acc"),
            "epochs_nums": summary.get("epochs_nums"),
            "total_time_sec": summary.get("total_time_sec"),
            "device": summary.get("device"),
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
            "run_name",
            "model",
            "augmentation",
            "lr",
            "batch_size",
            "optimizer",
            "scheduler",
            "epochs",
            "status",
            "best_epoch",
            "best_val_acc",
            "lastEpoch_val_acc",
            "epochs_nums",
            "total_time_sec",
            "device",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    print("\nFinished all experiments.")
    print(f"Saved to: {results_csv}")


if __name__ == "__main__":
    main()