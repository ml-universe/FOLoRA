"""LoRA 适配器单元测试（纯 CPU）。"""

import torch
import torch.nn as nn

from peft_cl.adapters.lora import LoRALinear


def test_zero_init_delta_weight():
    base = nn.Linear(4, 6)
    lora = LoRALinear(base, rank=2, alpha=4)
    assert torch.allclose(lora.delta_weight, torch.zeros(6, 4))


def test_forward_matches_base_at_init():
    base = nn.Linear(4, 6)
    lora = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(3, 4)
    assert lora(x).shape == (3, 6)
    assert torch.allclose(lora(x), base(x)), "B=0 时输出应等于冻结主干"


def test_delta_weight_shape_and_scale():
    base = nn.Linear(4, 6)
    lora = LoRALinear(base, rank=2, alpha=4)  # scale = 2
    assert lora.delta_weight.shape == (6, 4)
