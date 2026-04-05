# Dogs vs. Cats Workflow

## 1. Dataset Layout

Use the teacher-provided dataset with the following structure:

```text
data/datasets/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

The label mapping is:

- `cat = 0`
- `dog = 1`

## 2. Environment Setup

```bash
conda activate ee6483-dogcat
pip install torch torchvision numpy pandas matplotlib tqdm pillow scikit-learn
sudo apt-get update && sudo apt-get install -y tree
```

## 3. Check Dataset

```bash
tree -L 3 data/datasets
```

Optional quick check:

```bash
python - <<'PY'
from utils.dataset import DatasetConfig, build_dataloaders
config = DatasetConfig(root_dir="./data/datasets", img_size=224, batch_size=8, num_workers=0, augmentation="none")
train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(config)
print("class_to_idx =", class_to_idx)
print("train batches =", len(train_loader))
print("val batches =", len(val_loader))
batch = next(iter(test_loader))
print("test ids =", batch[1][:5])
PY
```

## 4. Train Model

### Debug run

```bash
python train.py \
  --data_root ./data/datasets \
  --model cnn \
  --epochs 2 \
  --batch_size 16 \
  --lr 0.001 \
  --augmentation light \
  --train_subset 200 \
  --val_subset 100 \
  --output_dir outputs \
  --run_name debug_run
```

### Main training run

```bash
python train.py \
  --data_root ./data/datasets \
  --model resnet \
  --pretrained \
  --unfreeze_layer4 \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.0001 \
  --optimizer adamw \
  --scheduler plateau \
  --output_dir outputs \
  --run_name resnet_run
```

Training outputs are saved in:

```text
outputs/resnet_run/
├── acc_curve.png
├── loss_curve.png
├── model_best.pth
└── summary.json
```

## 5. Generate Submission

```bash
python predict.py \
  --data_root ./data/datasets \
  --model resnet \
  --checkpoint outputs/resnet_run/model_best.pth \
  --pretrained \
  --unfreeze_layer4 \
  --output_csv submission.csv
```

Check the result:

```bash
head submission.csv
wc -l submission.csv
```

## 6. Evaluate on Validation Set

```bash
PYTHONPATH=. python tools/evaluate_val.py \
  --data_root ./data/datasets \
  --model resnet \
  --checkpoint outputs/resnet_run/model_best.pth \
  --pretrained \
  --unfreeze_layer4 \
  --output_json outputs/resnet_run/val_metrics.json
```

This generates:

```text
outputs/resnet_run/val_metrics.json
```

## 7. Visualize Misclassified Validation Samples

```bash
PYTHONPATH=. python tools/visualize_errors.py \
  --data_root ./data/datasets \
  --model resnet \
  --checkpoint outputs/resnet_run/model_best.pth \
  --pretrained \
  --unfreeze_layer4 \
  --output_png outputs/resnet_run/val_errors.png
```

This generates:

```text
outputs/resnet_run/val_errors.png
```

## 8. Final Files for Submission / Analysis

Main output files:

```text
submission.csv
outputs/resnet_run/summary.json
outputs/resnet_run/val_metrics.json
outputs/resnet_run/val_errors.png
outputs/resnet_run/acc_curve.png
outputs/resnet_run/loss_curve.png
```
