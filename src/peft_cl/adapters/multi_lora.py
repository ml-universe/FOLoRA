"""多任务 LoRA：每个任务一个 (A_t, B_t)，前向累加 —— O-LoRA 基线专用。

与单 LoRA 的区别：O-LoRA 认为每个任务应占据正交的低秩子空间，因此给每个任务
分配独立的 A/B，推理时把所有任务的增量加起来 ΔW=Σ_t ΔW_t。
"""

import math

import torch
import torch.nn as nn


class _Adapter(nn.Module):
    """一个任务的低秩增量 (lora_A: in->rank, lora_B: rank->out)。"""

    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, rank, bias=False)   # 权重 (rank, in)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)  # 权重 (out, rank)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)


class MultiLoRALinear(nn.Module):
    """包装冻结的 nn.Linear，持有按任务累积的多个低秩增量。

    `aggregate` 控制推理时怎么合并这些增量：
      - "sum" （默认，O-LoRA 论文原式）：ΔW = Σ_i B_i A_i，累加不归一。
      - "mean"：ΔW = (1/T) Σ_i B_i A_i，用于**公平性对照**。
    原式下 ΔW 的秩上限随任务数线性增长（rank 16 × 20 任务 = 320 维 / 768 维输入 ≈ 42%），
    在 PTM + NCM 协议下会把特征推歪；mean 是「换个聚合方式能不能救回来」的对照。
    两种都报，用来挡掉「你的 O-LoRA 复现是坏的」这个质疑。
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int, aggregate: str = "sum"):
        super().__init__()
        if aggregate not in ("sum", "mean"):
            raise ValueError(f"未知的 olora_aggregate: {aggregate!r}（只支持 sum / mean）")
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.aggregate = aggregate
        out_dim, in_dim = base.weight.shape
        self.in_dim, self.out_dim = in_dim, out_dim
        self.adapters = nn.ModuleList()  # 每任务一个 _Adapter

    def add_task(self) -> _Adapter:
        """为下一个任务新增一个 adapter 并返回（A 需由调用方做正交初始化）。

        新 adapter 需与主干同设备/同 dtype（训练中才调用，此时模型已在 GPU）。
        """
        a = _Adapter(self.in_dim, self.out_dim, self.rank)
        a = a.to(self.base.weight.device, self.base.weight.dtype)
        self.adapters.append(a)
        return a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.adapters:
            return out
        scale = self.scale / len(self.adapters) if self.aggregate == "mean" else self.scale
        for a in self.adapters:
            out = out + scale * a.lora_B(a.lora_A(x))
        return out


def inject_multi_lora(model: nn.Module, rank: int, alpha: int,
                      targets=("mlp.3",), aggregate: str = "sum"):
    """把 model 中名字以 targets 结尾的 nn.Linear 替换为 MultiLoRALinear。

    默认注入 mlp.3（同 inject_lora，不能注入 out_proj，见其 docstring）。
    `aggregate` 透传给 MultiLoRALinear（"sum" = 论文原式，"mean" = 公平性对照）。
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in targets):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            replacements.append((parent, parts[-1], module))
    injected = []
    for parent, attr, linear in replacements:
        m = MultiLoRALinear(linear, rank, alpha, aggregate=aggregate)
        setattr(parent, attr, m)
        injected.append(m)
    return injected


def iter_multi_lora(model: nn.Module):
    """遍历模型中所有 MultiLoRALinear 模块。"""
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            yield module
