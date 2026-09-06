"""数据层：数据集加载 + 类增量切分。"""
from .datasets import load_cifar
from .split import ContinualSplit, make_class_order

__all__ = ["load_cifar", "ContinualSplit", "make_class_order"]
