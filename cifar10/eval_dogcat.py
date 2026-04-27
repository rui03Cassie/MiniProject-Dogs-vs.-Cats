"""
eval_dogcat.py — Find correct and misclassified samples from the Dogs vs. Cats
validation set for case study analysis (report section f).

Place this file in the project root:
  /projects/projectsLC/MiniProject-Dogs-vs.-Cats/eval_dogcat.py

Outputs saved to outputs/case_study/:
  - cases_correct.png
  - cases_wrong_top4.png
  - cases_wrong.png
  - case_study.json
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Always add project root to sys.path so `from models import build_model` works
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
print(f"[INFO] Project root: {_ROOT}")
print(f"[INFO] sys.path[0]: {sys.path[0]}")

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision import transforms, datasets
    from models import build_model

    # ── Config ────────────────────────────────────────────────────────────────
    CKPT = (
        "/projects/projectsLC/MiniProject-Dogs-vs.-Cats/outputs/"
        "grid_search_fixed_ep50_bs128/"
        "resnet_aug-strong_lr-0.0001_bs-128_opt-adamw_sch-cosine_ep-50/"
        "model_best.pth"
    )
    VAL_DIR    = "/projects/projectsLC/MiniProject-Dogs-vs.-Cats/data/datasets/val"
    OUTPUT_DIR = "/projects/projectsLC/MiniProject-Dogs-vs.-Cats/outputs/case_study"
    IMG_SIZE   = 224
    BATCH      = 64
    SEED       = 42

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Transforms ────────────────────────────────────────────────────────────
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading val set from: {VAL_DIR}")
    dataset     = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
    raw_dataset = datasets.ImageFolder(VAL_DIR)   # PIL images, no transform
    class_names = dataset.classes                  # ['cat', 'dog']
    print(f"[INFO] Classes: {class_names},  Val size: {len(dataset)}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading checkpoint: {CKPT}")
    model = build_model("resnet", num_classes=2, dropout=0.3,
                        pretrained=False, unfreeze_layer4=True)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.to(device).eval()
    print("[INFO] Model loaded.")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs  = F.softmax(logits, dim=1).cpu()
            preds  = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    acc = (all_preds == all_labels).mean()
    print(f"[INFO] Val accuracy: {acc:.4f}")

    correct_idx = list(np.where(all_preds == all_labels)[0])
    wrong_idx   = list(np.where(all_preds != all_labels)[0])
    print(f"[INFO] Correct: {len(correct_idx)},  Wrong: {len(wrong_idx)}")

    correct_idx.sort(key=lambda i: -all_probs[i][all_preds[i]])
    wrong_idx.sort(  key=lambda i:  all_probs[i][all_preds[i]])

    # ── Plot helper ───────────────────────────────────────────────────────────
    def plot_samples(indices, title, save_path, ncols=4):
        if len(indices) == 0:
            print(f"[WARN] No samples to plot for: {title}")
            return
        nrows = max(1, (len(indices) + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.5 * ncols, 3.8 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)
        fig.suptitle(title, fontsize=12, fontweight="bold")

        for k, idx in enumerate(indices):
            r, c = divmod(k, ncols)
            ax   = axes[r][c]
            img, _ = raw_dataset[idx]
            ax.imshow(img)
            true_cls = class_names[int(all_labels[idx])]
            pred_cls = class_names[int(all_preds[idx])]
            conf     = float(all_probs[idx][all_preds[idx]])
            is_ok    = (all_preds[idx] == all_labels[idx])
            ax.set_title(
                f"True: {true_cls}\nPred: {pred_cls}\nConf: {conf:.3f}",
                fontsize=8, color="green" if is_ok else "red",
            )
            ax.axis("off")

        for k in range(len(indices), nrows * ncols):
            r, c = divmod(k, ncols)
            axes[r][c].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved: {save_path}")

    # ── Output ────────────────────────────────────────────────────────────────
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    top_cat = [i for i in correct_idx if all_labels[i] == 0][:2]
    top_dog = [i for i in correct_idx if all_labels[i] == 1][:2]
    plot_samples(top_cat + top_dog,
                 "Correctly Classified Samples (high confidence)",
                 out / "cases_correct.png", ncols=4)

    plot_samples(wrong_idx[:4],
                 "Most Borderline Misclassified Samples",
                 out / "cases_wrong_top4.png", ncols=4)

    plot_samples(wrong_idx,
                 f"All Misclassified Samples (n={len(wrong_idx)})",
                 out / "cases_wrong.png", ncols=8)

    wrong_meta = []
    for idx in wrong_idx:
        img_path, _ = raw_dataset.imgs[idx]
        wrong_meta.append({
            "index":      int(idx),
            "path":       str(img_path),
            "true_label": class_names[int(all_labels[idx])],
            "pred_label": class_names[int(all_preds[idx])],
            "confidence": float(all_probs[idx][all_preds[idx]]),
        })

    with open(out / "case_study.json", "w") as f:
        json.dump({
            "overall_accuracy": float(acc),
            "n_correct": int(len(correct_idx)),
            "n_wrong":   int(len(wrong_idx)),
            "misclassified": wrong_meta,
        }, f, indent=2)
    print(f"[INFO] Saved: {out / 'case_study.json'}")
    print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)