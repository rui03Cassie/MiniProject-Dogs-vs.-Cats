from __future__ import annotations
import argparse
from pathlib import Path
import math
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.dataset import DatasetConfig, build_dataloaders
from models import build_model


@torch.no_grad()
def collect_errors(model, loader, device, max_errors=12):
    model.eval()
    errors = []

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        wrong_mask = preds != labels
        wrong_idx = wrong_mask.nonzero(as_tuple=False).flatten().tolist()

        for i in wrong_idx:
            errors.append((
                images[i].cpu(),
                int(labels[i].cpu()),
                int(preds[i].cpu()),
            ))
            if len(errors) >= max_errors:
                return errors
    return errors


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.clone() * std + mean
    return img.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "resnet", "resnet34"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--normalize_mode", type=str, default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--unfreeze_layer4", action="store_true")
    parser.add_argument("--max_errors", type=int, default=12)
    parser.add_argument("--output_png", type=str, default="outputs/resnet_run/val_errors.png")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = DatasetConfig(
        root_dir=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation="none",
        normalize_mode=args.normalize_mode,
        seed=42,
    )

    _, val_loader, _, _ = build_dataloaders(config)

    model = build_model(
        args.model,
        2,
        args.dropout,
        args.pretrained,
        args.train_backbone,
        args.unfreeze_layer4,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    errors = collect_errors(model, val_loader, device, args.max_errors)
    if not errors:
        print("No misclassified samples found.")
        raise SystemExit(0)

    class_names = {0: "cat", 1: "dog"}
    n = len(errors)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, (img, true_label, pred_label) in enumerate(errors, start=1):
        plt.subplot(rows, cols, idx)
        img = denormalize(img).permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"true={class_names[true_label]}\npred={class_names[pred_label]}")
        plt.axis("off")

    plt.tight_layout()
    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved error visualization to: {out}")


if __name__ == "__main__":
    main()
