# Dogs vs. Cats Classification Training part
This repository is used for constructing, training, and generating prediction results for the three cat-dog classification models. 

Scope:
- supervised image classification on dog/cat images
- training on the training split and validation on the validation split
- prediction on the test split
- exporting the final test predictions into `submission.csv`
- comparing different models and training settings

The current codebase supports both a **custom CNN baseline** and **transfer-learning-based residual networks(ResNet18 and ResNet34)**

## 1. Environment Setup

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm
- pillow
- scikit-learn

### Install dependencies

```bash
pip install torch torchvision numpy pandas matplotlib tqdm pillow scikit-learn
```

If you use NVIDIA GPU, install the CUDA-enabled PyTorch build that matches your local CUDA version.

**Verify GPU**

```python
import torch
print(torch.cuda.is_available())      # True if GPU is available
print(torch.cuda.get_device_name(0))  # e.g. NVIDIA A100
```

> The training scripts auto-detect device priority:  **CUDA → MPS (Apple Silicon) → CPU**.

---

## 2. Data Preparation
### 2.1 Expected Input
The training pipeline assumes the dataset has already been organized as:

```text
data/
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

### 2.2 If the original dataset is still flat:
Run:

```bash
python utils/split_dataset.py `
  --root_dir ./dogs-vs-cats `
  --val_ratio 0.2 `
  --seed 42 `
  --copy `
  --force_clean
```

## 3. Core files

- `models.py`  
  Defines the custom CNN, ResNet-18, and ResNet-34 models.

- `train.py`  
  Main training script. Handles training, validation, checkpoint saving, early stopping, and curve plotting.

- `predict.py`  
  Loads the best checkpoint and generates `submission.csv` for the test set.

### 3.1. train.py
#### Purpose
Train classification models on the Dogs vs. Cats dataset and automatically save the model with the best validation performance.

Support:
- Custom CNN trained from scratch
- Transfer learning with ResNet18
- Transfer learning with ResNet34
- Hyperparameter tuning through command-line arguments
- Various optimizers, augmentation settings, and learning rate schedulers
- Automatic early stopping
- Automatic generation of loss and accuracy curves
- Reproducible by fixing the random seed

#### Input
- A pre-split dataset directory containing `train/`, `val/`, and `test/`
- Model type:
  - `cnn`
  - `resnet`
  - `resnet34`
- Adjustable hyperparameters:
  - augmentation
  - learning rate
  - batch size
  - optimizer
  - scheduler
  - early stopping patience
  - early stopping minimum delta
  - dropout
  - pretrained / unfreeze settings

#### Output
After training is completed, files will be generated under the `outputs/` directory:

```bash
outputs/
└── run_name/
    ├── model_best.pth
    ├── summary.json
    ├── loss_curve_lr-..._aug-....png
    └── acc_curve_lr-..._aug-....png
```
#### Notes
- `model_best.pth` stores the checkpoint with the best validation accuracy.
- `summary.json` records the best epoch, best validation accuracy, last validation accuracy, device, runtime, and configuration.
- Early stopping is triggered based on **validation loss**, while the best model checkpoint is saved according to **validation accuracy**.
- The loss function is `CrossEntropyLoss(label_smoothing=0.05)`, where label_smoothing helps to improve generalization and reduce overconfidence in the predictions.

#### How to Run

##### Train a custom CNN
```bash
python train.py `
  --data_root ./data/datasets `
  --model cnn `
  --epochs 10 `
  --batch_size 32 `
  --lr 0.001 `
  --augmentation light `
  --optimizer adam `
  --scheduler plateau `
  --dropout 0.3 `
  --output_dir outputs `
  --run_name cnn_run
```

##### Train ResNet18 with transfer learning
```bash
python train.py `
  --data_root ./data/datasets `
  --model resnet `
  --pretrained `
  --unfreeze_layer4 `
  --epochs 10 `
  --batch_size 32 `
  --lr 0.0001 `
  --augmentation strong `
  --optimizer adamw `
  --scheduler plateau `
  --dropout 0.3 `
  --output_dir outputs `
  --run_name resnet18_run
```

##### Train ResNet34 with transfer learning
```bash
python train.py `
  --data_root ./data/datasets `
  --model resnet34 `
  --pretrained `
  --unfreeze_layer4 `
  --epochs 10 `
  --batch_size 32 `
  --lr 0.0001 `
  --augmentation strong `
  --optimizer adamw `
  --scheduler cosine `
  --dropout 0.3 `
  --output_dir outputs `
  --run_name resnet34_run
```

---

### 3.2. `models.py`
#### Purpose
Define and build the classification models used in this project. This file provides a unified interface to construct different network architectures for training, evaluation, and inference.

Support:
- Custom CNN
- ResNet18 transfer learning
- ResNet34 transfer learning
- Adjustable dropout in the classifier head
- Flexible freezing / unfreezing of pretrained backbones
- Unified `build_model(...)` function

#### Input
- Model type:
  - `cnn`
  - `resnet (refers to resnet18)`
  - `resnet34`
- Number of classes
- Dropout rate
- Whether to use pretrained weights
- Whether to train the full backbone
- Whether to unfreeze only `layer4`

#### Output
Returns a PyTorch model instance that can be used directly in training or inference:

Main models and functions:
- `CNN`
- `ResNet`
- `build_model(model_name, num_classes, dropout, pretrained, train_backbone, unfreeze_layer4)`

#### Models Used

##### CNN
A custom convolutional neural network built from scratch that contains:
- Five progressive convolutional stages
- `Conv + BatchNorm + ReLU` blocks
- Max pooling for spatial downsampling: `MaxPool2d`
- `AdaptiveAvgPool2d((1,1))`
- A classifier head with:
  - `Flatten`
  - `Dropout`
  - `Linear(512 → 256)`
  - `ReLU`
  - `Dropout`
  - `Linear(256 → 2)`

##### ResNet
A transfer learning architecture based on pretrained torchvision ResNet backbones.
- `resnet` refers to **ResNet18**
- `resnet34` refers to **ResNet34**
- The original fully connected layer is replaced with a custom classifier head:
  - `Dropout`
  - `Linear(in_features → 256)`
  - `ReLU`
  - `Dropout`
  - `Linear(256 → 2)`

#### Notes
- By default, all backbone parameters are frozen.
- You can then choose one of the following fine-tuning strategies:
  - train the full backbone with `train_backbone=True`, or
  - unfreeze only `layer4` with `unfreeze_layer4=True`
- Both options cannot be enabled at the same time.
- This file is reused by `train.py`, `predict.py`, `evaluate_val.py`, and `visualize_errors.py`.

---

### 3.3. `predict.py`

#### Purpose
Load a trained checkpoint, generate predictions on the test set, then export the results as a submission CSV file.

#### Input
- A pre-split dataset directory containing `test/`
- A trained checkpoint file, usually `model_best.pth`
- Model type:
  - `cnn`
  - `resnet`
  - `resnet34`
- Inference settings such as:
  - image size
  - batch size
  - number of workers
  - normalization mode
  - dropout
  - pretrained / unfreeze settings

#### Output
A CSV file for submission:

```bash
submission.csv
├── id
└── label (0 = cat, 1 = dog)
```

Example output:

```bash
outputs/
└── submission_best.csv
```

#### Notes
- The script reconstructs the same model architecture used during training.
- It loads the trained checkpoint and performs deterministic inference on the test set.
- Predictions are sorted by image ID before saving.
- Label mapping:
  - `0 = cat`
  - `1 = dog`

#### How to Run

##### Generate submission with CNN
```bash
python predict.py `
  --data_root ./data/datasets `
  --model cnn `
  --checkpoint outputs/cnn_run/model_best.pth `
  --batch_size 32 `
  --dropout 0.3 `
  --output_csv outputs/cnn_submission.csv
```

##### Generate submission with ResNet18
```bash
python predict.py `
  --data_root ./data/datasets `
  --model resnet `
  --checkpoint outputs/resnet18_run/model_best.pth `
  --batch_size 32 `
  --dropout 0.3 `
  --pretrained `
  --unfreeze_layer4 `
  --output_csv outputs/resnet18_submission.csv
```

##### Generate submission with ResNet34
```bash
python predict.py `
  --data_root ./data/datasets `
  --model resnet34 `
  --checkpoint outputs/resnet34_run/model_best.pth `
  --batch_size 32 `
  --dropout 0.3 `
  --pretrained `
  --unfreeze_layer4 `
  --output_csv outputs/resnet34_submission.csv
```

---

## 4. Recommended Workflow

### 4.1. Debug

```bash
python train.py `
  --data_root ./data/datasets `
  --model cnn `
  --epochs 2 `
  --batch_size 16 `
  --lr 0.0001 `
  --augmentation light `
  --train_subset 200 `
  --val_subset 100 `
  --dropout 0.3 `
  --early_stop_patience 3 `
  --early_stop_min_delta 0.0005 `
  --output_dir outputs `
  --run_name debug_run
```

### 4.2. Formal Training
Augmentation is set to light as it is the most reasonable compromise.
Dropout rate is set to `0.3`, as a moderate level of regularization is generally helpful for improving generalization. Although the 50-epoch dropout comparison in the report is only conducted on ResNet34, the final runs of the ResNet-based models also suggest that an appropriate amount of regularization is beneficial.
Early stopping patience is set to `3` for short training runs (e.g., 10 epochs), but set to `7` for longer runs (e.g., 50 epochs in grid search).


#### Train CNN

```bash
python train.py `
  --data_root ./data/datasets `
  --model cnn `
  --epochs 10 `
  --batch_size 32 `
  --lr 0.0001 `
  --augmentation light `
  --dropout 0.3 `
  --optimizer adamw `
  --scheduler plateau `
  --early_stop_patience 3 `
  --early_stop_min_delta 0.0005 `
  --output_dir outputs `
  --run_name cnn_run1
```

#### Train ResNet18

```bash
python train.py `
  --data_root ./data/datasets `
  --model resnet `
  --pretrained `
  --unfreeze_layer4 `
  --epochs 10 `
  --batch_size 32 `
  --lr 0.0001 `
  --augmentation light `
  --dropout 0.3 `
  --optimizer adamw `
  --scheduler plateau `
  --early_stop_patience 3 `
  --early_stop_min_delta 0.0005 `
  --output_dir outputs `
  --run_name resnet18_run
```

#### Train ResNet34

```bash
python train.py `
  --data_root ./data/datasets `
  --model resnet34 `
  --pretrained `
  --unfreeze_layer4 `
  --epochs 10 `
  --batch_size 32 `
  --lr 0.0001 `
  --augmentation light `
  --dropout 0.3 `
  --optimizer adamw `
  --scheduler plateau `
  --early_stop_patience 3 `
  --early_stop_min_delta 0.0005 `
  --output_dir outputs `
  --run_name resnet34_run
```

#### Hyperparameter Tuning with Grid Search

```bash
python .\tools\run_grid_search.py `
  --data_root ./data/datasets `
  --output_dir outputs/grid_search_fixed_ep50 `
  --models cnn resnet resnet34 `
  --augmentations none light strong `
  --lrs 0.001 0.0005 0.0001 `
  --batch_sizes 32 64 128 `
  --optimizers adam adamw `
  --schedulers plateau cosine `
  --dropouts 0.2 0.3 `
  --epochs_list 50 `
  --early_stop_patience 7 `
  --early_stop_min_delta 0.0005
```

### 4.3. Compare three models and corresponding settings

Check outputs/grid_search_fixed_ep50/grid_results.csv or outputs/grid_search_fixed_ep50/final_report.json

-> Choose the configuration with the highest validation accuracy (best_val_acc)

### 4.4. Generate the Final Submission file (Only once)
 Use the best-performing configuration (model type, augmentation, learning rate, batch size, optimizer, scheduler, and epochs) to generate the final submission.csv file.

#### If ResNet34 is the best (most of the case):

```bash
python predict.py `
  --data_root ./data/datasets `
  --model resnet34 `
  --checkpoint outputs/resnet34_run/model_best.pth `
  --pretrained `
  --unfreeze_layer4 `
  --dropout 0.3 `
  --output_csv submission.csv
```

#### If ResNet18 is the best:

```bash
python predict.py `
  --data_root ./data/datasets `
  --model resnet `
  --checkpoint outputs/resnet18_run/model_best.pth `
  --pretrained `
  --unfreeze_layer4 `
  --dropout 0.3 `
  --output_csv submission.csv
```

#### If CNN is the best:

```bash
python predict.py `
  --data_root ./data/datasets `
  --model cnn `
  --checkpoint outputs/cnn_run1/model_best.pth `
  --dropout 0.3 `
  --output_csv submission.csv
```