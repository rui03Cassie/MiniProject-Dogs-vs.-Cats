from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from utils.dataset import DatasetConfig, build_dataloaders
from models import build_model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


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
    parser.add_argument("--output_json", type=str, default="outputs/resnet_run/val_metrics.json")
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

    _, val_loader, _, class_to_idx = build_dataloaders(config)

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

    y_true, y_pred = evaluate(model, val_loader, device)

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred,
        target_names=["cat", "dog"],
        digits=4,
        output_dict=True
    )
    accuracy = report["accuracy"]

    result = {
        "accuracy": accuracy,
        "class_to_idx": class_to_idx,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
