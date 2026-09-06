"""FOLoRA 正则项单元测试（纯 CPU）：必须惩罚「变化量 δ = ΔW − ΔW_ref」而非「绝对量 ΔW」。

这个测试守护一个曾经踩过的坑：旧实现正则化绝对量 `delta_weight`，导致任务一开始
（ΔW 尚未变化）reg 就 > 0，把老任务已学到的方向一起往外推、等于主动抹掉老知识。
正确行为是「ΔW == ref_delta 时 reg 恒为 0，仅当 ΔW 偏离参考点才有惩罚」。
"""

import torch
import torch.nn as nn

from peft_cl.methods.folora import FOLoRAMethod
from peft_cl.utils.config import CLConfig


class MiniModel(nn.Module):
    """含 mlp.3 的迷你模型，供 inject_lora 注入（out=8, in=8）。"""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 8),   # mlp.3
        )

    def forward(self, x):
        return self.mlp(x)


def _trained_method():
    """构造一个「任务 0 已训练结束」状态的 FOLoRA：非零 ΔW + ref_delta 快照 + 保护子空间。"""
    model = MiniModel()
    cfg = CLConfig(method="folora", lora_rank=2, lora_alpha=2)
    m = FOLoRAMethod(model, cfg)

    # 模拟任务 0 训练后的非零 LoRA 权重
    with torch.no_grad():
        for lora in m.loras:
            lora.lora_B.weight.add_(0.3)
            lora.lora_A.weight.add_(0.2)

    # 任务 0 结束后：记录 ΔW 快照
    m.ref_delta = {i: lora.delta_weight.detach().clone()
                   for i, lora in enumerate(m.loras)}

    # 每个 LoRA 给一个保护方向（第一维），特征值 1
    for i, lora in enumerate(m.loras):
        out_dim = lora.lora_B.weight.shape[0]
        w = torch.ones(1)
        V = torch.zeros(out_dim, 1)
        V[0, 0] = 1.0
        m.protected[i] = (w, V)
    return m


def test_reg_zero_when_no_change():
    m = _trained_method()
    # ΔW 仍等于参考点 → 变化量为 0 → 正则必须为 0（旧实现这里会 > 0）
    assert m.regularization_loss().item() == 0.0


def test_reg_positive_when_delta_changes():
    m = _trained_method()
    with torch.no_grad():
        for lora in m.loras:
            lora.lora_B.weight.add_(0.5)   # 使 ΔW 偏离参考点
    assert m.regularization_loss().item() > 0.0
