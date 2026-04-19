from __future__ import annotations
import argparse
import random
import numpy as np
import torch
import os
from pathlib import Path
import pandas as pd
from utils.dataset import DatasetConfig, build_dataloaders
from models import build_model
from tqdm import tqdm


@torch.no_grad()
def predict_test(model, loader, device):
    model.eval()
    results = []

    process = tqdm(loader)
    for batch in process:
        images, image_ids, _ = batch
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for i in range(len(preds)):
            results.append({"id": int(image_ids[i]), "label": int(preds[i].cpu())})
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate submission.csv for Dogs vs Cats")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data/ directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "resnet", "resnet34"])

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--normalize_mode", type=str, default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--unfreeze_layer4", action="store_true")
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    config = DatasetConfig(
        root_dir=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation="none",
        normalize_mode=args.normalize_mode,
        seed=args.seed,
    )

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(config)
    model = build_model(args.model,2, args.dropout, args.pretrained, args.train_backbone, args.unfreeze_layer4).to(device)
    print("class_to_idx=", class_to_idx)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    results = predict_test(model, test_loader, device)
    df = pd.DataFrame(results).sort_values("id").reset_index(drop=True)

    output_csv = Path(args.output_csv)
    save_folder = output_csv.parent
    save_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved submission to: {output_csv}")
    print(df.head())


if __name__ == "__main__":
    main()