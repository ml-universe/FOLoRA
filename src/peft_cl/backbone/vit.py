"""ViT 主干：加载 torchvision 预训练 ViT-B/16，冻结主干，替换为可扩展分类头。

类增量（CIL）需要一个「输出维度随已见类别增长」的分类头，因此把 torchvision
自带的 1000 类 ImageNet 头丢弃，换成 ExpandableHead。
"""

import torch
import torch.nn as nn
import torchvision.models as M


class ExpandableHead(nn.Module):
    """可扩展线性分类头：输出维度随已见类别数增长（类增量必需）。

    只维护一个 nn.Linear；扩充时保留旧类权重、新类权重随机初始化。
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.in_dim = in_dim
        self.head = nn.Linear(in_dim, num_classes)

    def expand(self, num_classes: int) -> None:
        """把输出维度扩到 num_classes，旧类权重/偏置原样保留。"""
        if num_classes <= self.head.out_features:
            return
        old = self.head
        new = nn.Linear(self.in_dim, num_classes,
                        device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:old.out_features].copy_(old.weight)
            new.bias[:old.out_features].copy_(old.bias)
        self.head = new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def build_vit(backbone: str = "vit_b_16", num_classes: int = 5,
              pretrained: bool = True):
    """构建冻结主干的 ViT + 可扩展分类头。

    返回 (model, head)：head 是 ExpandableHead 的引用，训练时按任务调用 expand。
    """
    if backbone != "vit_b_16":
        raise ValueError(f"unsupported backbone: {backbone}")
    weights = M.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = M.vit_b_16(weights=weights)

    # 1) 冻结全部参数（含即将被替换的旧 1000 类头）
    for p in model.parameters():
        p.requires_grad = False

    # 2) 替换分类头为可扩展线性头（新头参数默认可训练）
    in_dim = model.heads.head.in_features
    head = ExpandableHead(in_dim, num_classes)
    model.heads = head
    return model, head
