"""数据集加载。

CIFAR-100 走 ModelScope 镜像（parquet），因 cs.toronto.edu 在国内仅 ~8kB/s 不可用。
CIFAR-10 暂走 torchvision（如需加速可同样加 ModelScope 镜像）。
"""

from torchvision import datasets, transforms

from .cifar_parquet import load_cifar100_parquet
from .imagenetr import load_imagenetr

# ImageNet 统计量：ViT 是 ImageNet 预训练，必须用同款归一化
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(image_size: int, train: bool) -> transforms.Compose:
    """构建图像变换：resize 到 224 + ImageNet 归一化，训练额外加随机水平翻转。"""
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def load_cifar(name: str, root: str, image_size: int):
    """加载 CIFAR-10/100，返回 (train_dataset, test_dataset)。

    返回的 dataset 保留原始类别标签（存在 .targets 里），类增量切分由 split.py 负责。
    """
    if name == "cifar100":
        return load_cifar100_parquet(root, image_size, get_transform)
    if name == "imagenetr":
        return load_imagenetr(root, image_size, get_transform)
    if name == "cifar10":
        ds_cls = datasets.CIFAR10
    else:
        raise ValueError(f"unknown dataset: {name}")
    train = ds_cls(root=root, train=True, download=True,
                   transform=get_transform(image_size, True))
    test = ds_cls(root=root, train=False, download=True,
                  transform=get_transform(image_size, False))
    return train, test
