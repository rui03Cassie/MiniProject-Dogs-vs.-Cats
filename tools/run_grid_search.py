from __future__ import annotations
import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def build_common_model_args(model: str) -> list[str]:
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model}")
    return MODEL_CONFIGS[model]["extra_args"]


def evaluate_one_run(
    data_root: str,
    run_dir: Path,
    run_name: str,
    model: str,
    batch_size: int,
    num_workers: int,
):
    checkpoint = run_dir / "model_best.pth"
    if not checkpoint.exists():
        return False

    common_args = build_common_model_args(model)

    eval_cmd = [
        sys.executable, "tools/evaluate_val.py",
        "--data_root", data_root,
        "--model", model,
        "--checkpoint", str(checkpoint),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--output_json", str(run_dir / "val_metrics.json"),
    ] + common_args

    code = run_command(eval_cmd)
    return code == 0


def visualize_one_run(
    data_root: str,
    run_dir: Path,
    model: str,
    batch_size: int,
    num_workers: int,
):
    checkpoint = run_dir / "model_best.pth"
    if not checkpoint.exists():
        return False

    common_args = build_common_model_args(model)

    vis_cmd = [
        sys.executable, "tools/visualize_errors.py",
        "--data_root", data_root,
        "--model", model,
        "--checkpoint", str(checkpoint),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--output_png", str(run_dir / "val_errors.png"),
    ] + common_args

    code = run_command(vis_cmd)
    return code == 0


def predict_one_run(
    data_root: str,
    run_dir: Path,
    model: str,
    batch_size: int,
    num_workers: int,
    output_csv: Path,
):
    checkpoint = run_dir / "model_best.pth"
    if not checkpoint.exists():
        return False

    common_args = build_common_model_args(model)

    pred_cmd = [
        sys.executable, "predict.py",
        "--data_root", data_root,
        "--model", model,
        "--checkpoint", str(checkpoint),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--output_csv", str(output_csv),
    ] + common_args

    code = run_command(pred_cmd)
    return code == 0


def pick_best(rows, key_name="best_val_acc"):
    valid = [r for r in rows if r.get("status") == "ok" and r.get(key_name) is not None]
    if not valid:
        return None
    valid = sorted(valid, key=lambda x: (-float(x[key_name]), x["run_name"]))
    return valid[0]


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
    parser.add_argument("--optimizers", nargs="*", default=["adamw"])
    parser.add_argument("--schedulers", nargs="*", default=["plateau"])
    parser.add_argument("--epochs_list", nargs="*", type=int, default=[10, 15])

    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--val_subset", type=int, default=None)

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_visualize", action="store_true")
    parser.add_argument("--skip_predict_best", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "grid_results.csv"
    final_report_path = output_dir / "final_report.json"
    best_submission_path = output_dir / "submission_best.csv"

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
        val_metrics_path = run_dir / "val_metrics.json"

        if summary_path.exists() and not args.force:
            print(f"Skip existing run: {run_name}")
            summary = load_json(summary_path)
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

            summary = load_json(summary_path)

        if not args.skip_eval:
            if args.force or (not val_metrics_path.exists()):
                evaluate_one_run(
                    data_root=args.data_root,
                    run_dir=run_dir,
                    run_name=run_name,
                    model=model,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                )

        val_metrics = load_json(val_metrics_path)

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
            "best_val_acc": safe_float(summary.get("best_val_acc")),
            "lastEpoch_val_acc": safe_float(summary.get("lastEpoch_val_acc")),
            "epochs_nums": summary.get("epochs_nums"),
            "total_time_sec": summary.get("total_time_sec"),
            "device": summary.get("device"),
            "eval_accuracy": safe_float(val_metrics.get("accuracy")) if val_metrics else None,
            "val_metrics_json": str(val_metrics_path) if val_metrics_path.exists() else "",
        }
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            x.get("status") != "ok",
            -(x.get("best_val_acc") if x.get("best_val_acc") is not None else -1),
            x["run_name"],
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
            "eval_accuracy",
            "val_metrics_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    best_overall = pick_best(rows_sorted, "best_val_acc")
    best_per_model = {}
    for m in args.models:
        cand = [r for r in rows_sorted if r.get("model") == m]
        best_per_model[m] = pick_best(cand, "best_val_acc")

    best_per_aug = {}
    for aug in args.augmentations:
        cand = [r for r in rows_sorted if r.get("augmentation") == aug]
        best_per_aug[aug] = pick_best(cand, "best_val_acc")

    if not args.skip_visualize:
        chosen = []
        if best_overall:
            chosen.append(best_overall["run_name"])
        for item in best_per_model.values():
            if item:
                chosen.append(item["run_name"])
        chosen = sorted(set(chosen))

        run_lookup = {r["run_name"]: r for r in rows_sorted}
        for run_name in chosen:
            r = run_lookup[run_name]
            run_dir = output_dir / run_name
            visualize_one_run(
                data_root=args.data_root,
                run_dir=run_dir,
                model=r["model"],
                batch_size=r["batch_size"],
                num_workers=args.num_workers,
            )

    submission_generated = False
    if best_overall and not args.skip_predict_best:
        run_dir = output_dir / best_overall["run_name"]
        submission_generated = predict_one_run(
            data_root=args.data_root,
            run_dir=run_dir,
            model=best_overall["model"],
            batch_size=best_overall["batch_size"],
            num_workers=args.num_workers,
            output_csv=best_submission_path,
        )

    final_report = {
        "num_total_runs": len(rows_sorted),
        "num_success_runs": len([r for r in rows_sorted if r.get("status") == "ok"]),
        "best_overall": best_overall,
        "best_per_model": best_per_model,
        "best_per_augmentation": best_per_aug,
        "results_csv": str(results_csv),
        "submission_best_csv": str(best_submission_path) if submission_generated else None,
        "top10_runs": rows_sorted[:10],
    }

    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\nFinished all experiments.")
    print(f"Saved summary table to: {results_csv}")
    print(f"Saved final report to: {final_report_path}")
    if submission_generated:
        print(f"Saved best submission to: {best_submission_path}")


if __name__ == "__main__":
    main()
