"""
plot_cifar10_comparison.py — 绘制四组实验的val_acc对比曲线

Usage:
    python cifar10/plot_cifar10_comparison.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# ── 配置：实验路径和显示名称 ──────────────────────────────
EXPERIMENTS = [
    ("outputs/cifar10/exp1_baseline",           "Baseline (50k, no imbalance)"),
    ("outputs/cifar10/exp4_imbalanced_baseline", "Imbalanced baseline (34k, no fix)"),
    ("outputs/cifar10/exp2_class_weight",        "Class weight (34k)"),
    ("outputs/cifar10/exp3_oversample",          "Oversample (34k)"),
]
OUTPUT_PATH = "outputs/cifar10/comparison_val_acc.png"


def load_history(exp_dir: str):
    """从summary.json旁边读取history，若无则从summary重建单点"""
    history_path = Path(exp_dir) / "history.json"
    summary_path = Path(exp_dir) / "summary.json"

    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)["val_acc"]

    # fallback：只有summary时用best_val_acc画一个点（不画曲线）
    with open(summary_path) as f:
        summary = json.load(f)
    return None, summary  # signal to caller


def main():
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3", "#9E9E9E", "#FF9800", "#4CAF50"]
    linestyles = ["-", "--", "-.", ":"]

    for (exp_dir, label), color, ls in zip(EXPERIMENTS, colors, linestyles):
        summary_path = Path(exp_dir) / "summary.json"
        history_path = Path(exp_dir) / "history.json"

        with open(summary_path) as f:
            summary = json.load(f)

        best_acc   = summary["best_val_acc"]
        best_epoch = summary["best_epoch"]
        epochs_run = summary["epochs_run"]

        if history_path.exists():
            with open(history_path) as f:
                val_acc = json.load(f)["val_acc"]
            epochs = list(range(1, len(val_acc) + 1))
            ax.plot(epochs, val_acc, color=color, linestyle=ls, linewidth=2,
                    label=f"{label}  (best={best_acc:.4f} @ ep{best_epoch})")
            ax.scatter([best_epoch], [best_acc], color=color, s=60, zorder=5)
        else:
            # No history file — plot a single marker with annotation
            ax.scatter([best_epoch], [best_acc], color=color, s=100,
                       marker="*", zorder=5,
                       label=f"{label}  (best={best_acc:.4f} @ ep{best_epoch})")
            ax.annotate(f"{best_acc:.4f}", (best_epoch, best_acc),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("CIFAR-10: Validation Accuracy — Imbalance Handling Comparison", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()