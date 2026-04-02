# training部分简要说明

## 环境依赖

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm
- pillow

安装 via pip:

```bash
pip install torch torchvision numpy pandas matplotlib tqdm pillow
```

## 1. train.py

## 功能

用于训练分类模型，并自动保存验证集表现最好的模型。

支持：

- 自建 CNN
- ResNet18 迁移学习
- 多种优化器 / 学习率策略
- 自动 early stopping
- 自动绘制训练曲线

### 输入

- 已处理好的数据目录（data/）
- 模型类型（cnn / resnet）
- 训练参数（epochs、lr、batch_size 等）

### 输出

训练完成后，在 outputs/ 目录下生成：

```bash
outputs/
└── run_name/
    ├── model_best.pth        # ⭐ 最优模型（用于预测）
    ├── model_last.pth
    ├── model_summary.json    # 记录训练结果
    ├── loss_curve.png
    └── acc_curve.png
```

## 如何运行

### 1) Debug（先测试是否能跑通）

```bash
python train.py \
  --data_root ./datasets \
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

### 2) 正式训练

#### CNN

```bash
python train.py \
  --data_root ./datasets \
  --model cnn \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.001 \
  --augmentation light \
  --output_dir outputs \
  --run_name cnn_run1
```

#### ResNet18（推荐）

```bash
python train.py \
  --data_root ./datasets \
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

## 2. models.py

## 功能

定义模型结构，并提供统一接口 build_model()。

包含模型:

- CNN（自建模型）：简单
- ResNet18（迁移学习）：准确率更高，默认冻结 backbone，只训练最后几层

## 可选参数：

- `--pretrained`：使用 ImageNet 预训练权重
- `--train_backbone`：训练整个网络（不推荐）
- `--unfreeze_layer4`：解冻最后一层（layer4）

## 调用方法

```python
model = build_model(
    model_name="cnn",  # 或 "resnet"
    num_classes=2,
    dropout=0.3,
    pretrained=True,
    train_backbone=False,
    unfreeze_layer4=True
)
```

--> 用于做模型对比实验

## 3. predict.py

## 功能

使用训练好的模型，对测试集进行预测，并生成 submission.csv（其中 0 = cat, 1 = dog）

处理流程：

- 加载 test 数据
- 加载训练好的模型
- 前向传播得到预测结果
- 使用 argmax 得到类别
- 保存为 CSV

## 输入

- 训练好的模型（.pth）
- 测试数据（test）

## 输出

```bash
id,label
1,0
2,1
3,0
...
```

## 如何运行

### ResNet（通常准确率更高）

```bash
python predict.py \
  --data_root ./datasets \
  --model resnet \
  --checkpoint outputs/resnet_run/model_best.pth \
  --pretrained \
  --unfreeze_layer4 \
  --output_csv submission.csv
```

### CNN

```bash
python predict.py \
  --data_root ./datasets \
  --model cnn \
  --checkpoint outputs/cnn_run1/model_best.pth \
  --output_csv submission.csv
```

## 推荐运行流程

### 1. Debug（先测试是否能跑通）

```bash
python train.py \
  --data_root ./datasets \
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

### 2. 正式训练

#### 2.1. CNN

```bash
python train.py \
  --data_root ./datasets \
  --model cnn \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.001 \
  --augmentation light \
  --output_dir outputs \
  --run_name cnn_run1
```

#### 2.2. ResNet18（推荐）

```bash
python train.py \
  --data_root ./datasets \
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

### 3. 比较两个模型：

查看训练日志中的 val_acc 或 outputs/.../model_summary.json

-> 选择 验证集 accuracy 更高的模型

### 4. 用最优模型做预测（只做一次）

假设 ResNet 更好：

```bash
python predict.py \
  --data_root ./datasets \
  --model resnet \
  --checkpoint outputs/resnet_run/model_best.pth \
  --pretrained \
  --unfreeze_layer4 \
  --output_csv submission.csv
```

假设CNN更好：

```bash
python predict.py \
  --data_root ./datasets \
  --model cnn \
  --checkpoint outputs/cnn_run1/model_best.pth \
  --output_csv submission.csv
```