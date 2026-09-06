"""ImageNet-R 数据集加载（ModelScope 镜像，OpenDataLab/ImageNet-R）。

ImageNet-R：200 类、~3 万张域偏移图像（卡通/涂鸦/刺绣等），持续学习常用基准。
解压后为 200 个类别文件夹（按 synset id 命名），每类若干 jpg。

本模块把 200 类按每类 8:2 切成 train/test（固定 seed 可复现），
接口对齐 load_cifar，供 ContinualSplit 做类增量切分。
"""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageNetRDataset(Dataset):
    """从类别文件夹构建的数据集，含 .targets 属性（与 CIFAR 接口一致）。"""

    def __init__(self, samples, transform=None):
        self.samples = samples          # list of (path, label)
        self.targets = [lbl for _, lbl in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


def _collect(root: Path):
    """收集所有 (image_path, class_label)，类别按文件夹名排序编号 0..199。"""
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    samples = []
    for d in class_dirs:
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                samples.append((p, class_to_idx[d.name]))
    return samples


def load_imagenetr(root: str, image_size: int, transform_fn):
    """加载 ImageNet-R，返回 (train_dataset, test_dataset)。

    每类按 8:2 切 train/test（固定 seed=0 可复现）。
    """
    data_root = Path(root) / "imagenetr"
    if not (data_root / ".extracted").exists():
        raise FileNotFoundError(
            f"ImageNet-R 未解压：{data_root}。请先运行 `python -m scripts.download_imagenetr`。")
    # tar 内部有顶层 imagenet-r/ 文件夹，类文件夹在其下
    if (data_root / "imagenet-r").is_dir():
        data_root = data_root / "imagenet-r"

    samples = _collect(data_root)
    # 按类分组后每类 8:2 切分
    by_class = {}
    for p, lbl in samples:
        by_class.setdefault(lbl, []).append((p, lbl))
    rng = random.Random(0)
    train_samples, test_samples = [], []
    for lbl, items in by_class.items():
        rng.shuffle(items)
        n_train = int(len(items) * 0.8)
        train_samples.extend(items[:n_train])
        test_samples.extend(items[n_train:])

    train = ImageNetRDataset(train_samples, transform=transform_fn(image_size, True))
    test = ImageNetRDataset(test_samples, transform=transform_fn(image_size, False))
    return train, test
