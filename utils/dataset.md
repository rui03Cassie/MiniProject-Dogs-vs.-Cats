# dataset.py 简短说明

## 环境依赖
- Python 3.9+
- torch
- torchvision
- pillow

安装：
```bash
pip install torch torchvision pillow
```

## 数据准备流程
原始官方数据目录通常是：

```text
dogs-vs-cats/
├── train/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   └── dog.1.jpg
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

先运行 `split_dataset.py`，在原目录下生成新的 `data/`：

```text
dogs-vs-cats/
├── train/   # 原始数据，不动
├── test/    # 原始数据，不动
└── data/
    ├── train/
    │   ├── cat/
    │   └── dog/
    ├── val/
    │   ├── cat/
    │   └── dog/
    └── test/
```

## split_dataset.py 使用
```bash
python utils/split_dataset.py \
  --root_dir "/Users/maoyuhui/Desktop/NTU/courses/s2/EE6483/workplace/dogs-vs-cats" \
  --val_ratio 0.2 \
  --seed 42 \
  --copy \
  --force_clean
```

## 最基本使用
```python
from utils.dataset import DatasetConfig, build_dataloaders

config = DatasetConfig(
    root_dir="/Users/maoyuhui/Desktop/NTU/courses/s2/EE6483/workplace/dogs-vs-cats/data",
    img_size=224,
    batch_size=32,
    augmentation="light",
)

train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(config)
```

## 可切换的方法
### 1. `DatasetConfig(...)`
统一设置参数：
- `root_dir`: 已划分好的数据根目录，即 `data/`
- `img_size`: 输入尺寸
- `batch_size`: batch 大小
- `num_workers`: dataloader 线程数
- `augmentation`: `"none" | "light" | "strong"`
- `normalize_mode`: `"imagenet" | "none"`
- `train_subset`, `val_subset`: 子采样数量
- `seed`: 随机种子

### 2. `build_train_transform(...)`
构建训练集预处理 / 增强。

### 3. `build_eval_transform(...)`
构建验证集 / 测试集预处理。

### 4. `build_datasets(config)`
返回：
- `train_dataset`
- `val_dataset`
- `test_dataset`
- `class_to_idx`

### 5. `build_dataloaders(config)`
返回：
- `train_loader`
- `val_loader`
- `test_loader`
- `class_to_idx`

### 6. `describe_dataset(config)`
输出数据集摘要信息。

## augmentation 选项
- `none`: 仅基础预处理
- `light`: 轻量增强
- `strong`: 强增强，便于对比实验
