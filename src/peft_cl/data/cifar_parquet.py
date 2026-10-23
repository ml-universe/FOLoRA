"""从 ModelScope 镜像的 CIFAR-100 parquet 读取数据（绕开 cs.toronto.edu 慢速源）。

数据源：modelscope `cutedataset/cifar100`（HuggingFace cifar100 的镜像）。
格式：parquet 两列 —— img(bytes=无损 PNG，32×32) + fine_label(0-99 标准类标)。
本加载器只依赖 pyarrow + PIL，不依赖 modelscope 的 MsDataset（其与 datasets 库
版本不兼容会报错），直接读已下载的 parquet。
"""

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset


class CIFAR100Parquet(Dataset):
    """CIFAR-100 数据集（parquet 源），接口对齐 torchvision（含 .targets 属性）。

    __init__ 一次性把 PNG 预解码成 numpy 数组（N,32,32,3）uint8，训练时只做
    fromarray + transform，避免每个 epoch 反复解 PNG。
    """

    def __init__(self, parquet_path, transform=None):
        table = pq.read_table(parquet_path)
        imgs = table["img"].to_pylist()          # list of {'bytes': b'..', 'path': None}
        self.targets = table["fine_label"].to_pylist()  # list of int
        self.transform = transform
        self._images = np.stack([
            np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in imgs
        ])  # (N, 32, 32, 3) uint8

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = Image.fromarray(self._images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[i])


def load_cifar100_parquet(root, image_size, transform_fn):
    """加载 CIFAR-100（parquet 源）的 train/test，接口对齐 load_cifar。"""
    base = Path(root) / "cifar100"
    train_path = base / "train.parquet"
    test_path = base / "test.parquet"
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"CIFAR-100 parquet 未找到：{base}。请先运行 `python -m scripts.download_cifar` 下载。")
    train = CIFAR100Parquet(str(train_path), transform=transform_fn(image_size, True))
    test = CIFAR100Parquet(str(test_path), transform=transform_fn(image_size, False))
    return train, test
