"""LoRA 适配器：在冻结的 nn.Linear 上并接低秩增量 ΔW = (α/r)·B·A。

关键设计：forward 里对增量 delta 调 retain_grad()，使 loss.backward() 之后
delta.grad 保存 ∂L/∂delta —— 这是 FOLoRA 估计 Fisher 核所需的「输出方向梯度」。
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包装一个冻结的 nn.Linear，注入可训练的低秩增量。

    前向：y = W0·x + (α/r)·B·(A·x)。
    A 用 Kaiming 初始化、B 初始化为 0，保证初始 ΔW=0（训练从等价冻结主干起步）。
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        out_dim, in_dim = base.weight.shape
        # A: in->rank (权重 shape rank×in)，B: rank->out (权重 shape out×rank)
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        # A Kaiming 初始化、B 初始化为 0 → 初始 ΔW=0（训练从等价冻结主干起步）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self._last_delta = None  # 缓存本次 forward 的增量，供 Fisher 估计读取梯度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.lora_B(self.lora_A(x)) * self.scale
        if delta.requires_grad:
            delta.retain_grad()
        self._last_delta = delta
        return self.base(x) + delta

    @property
    def delta_weight(self) -> torch.Tensor:
        """低秩增量矩阵 ΔW = (α/r)·B·A，形状 (out_dim, in_dim)。"""
        return (self.lora_B.weight @ self.lora_A.weight) * self.scale


def inject_lora(model: nn.Module, rank: int, alpha: int,
                targets=("mlp.3",)) -> nn.Module:
    """把 model 中名字以 targets 结尾的 nn.Linear 全部替换为 LoRALinear。

    默认注入 MLP 下投影 mlp.3（输出维度 768，Fisher 核紧凑；每 block 1 个、共 12 个）。
    注意：不能注入注意力 out_proj —— torchvision 的 MultiheadAttention 用函数式接口
    直接读 out_proj.weight，LoRA 增量会被静默绕过；mlp.0/mlp.3 是普通 Sequential Linear，
    可正常包装。返回被注入的列表。
    """
    replacements = []  # (parent_module, attr_name, old_linear)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in targets):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            replacements.append((parent, parts[-1], module))

    injected = []
    for parent, attr, linear in replacements:
        lora = LoRALinear(linear, rank, alpha)
        setattr(parent, attr, lora)
        injected.append(lora)
    return injected


def iter_lora(model: nn.Module):
    """遍历模型中所有 LoRALinear 模块（训练器/方法计算正则项与重要性用）。"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module
