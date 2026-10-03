"""CODA-Prompt 基线（Smith et al. CVPR 2023）。

核心：冻结 ViT，维护一组 prompt 组件（N 个，每个长 L）+ 一组注意力 key。每输入用
query-key 注意力对组件做「软加权求和」，得到单个 prompt 拼到序列前（attend-then-
reduce）。相比 L2P 的硬 top-k，CODA 用注意力软选择、组件间可复用。这里复现其
注意力加权 + 冻结主干，属忠实简化版（未加正交正则）。

query 用 patches 的 mean pooling（冻结特征），keys 可学习，温度 τ 固定。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapters.prompt import PromptedEncoder
from .base import CLMethod


class CODASelector(nn.Module):
    def __init__(self, num_components: int, prompt_len: int, dim: int, tau: float = 0.1):
        super().__init__()
        self.prompt_len = prompt_len
        self.tau = tau
        self.prompt_components = nn.Parameter(
            torch.randn(num_components, prompt_len, dim) * 0.02)
        self.keys = nn.Parameter(torch.randn(num_components, dim) * 0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        q = patches.mean(dim=1)                                  # (B, D)
        attn = F.softmax(q @ self.keys.t() / self.tau, dim=-1)   # (B, N)
        # (B, N) @ (N, L*D) -> (B, L*D)，即按注意力加权求和组件
        flat = self.prompt_components.reshape(self.prompt_components.shape[0], -1)
        prompt = attn @ flat                                     # (B, L*D)
        return prompt.reshape(attn.shape[0], self.prompt_len, -1)  # (B, L, D)


class CODAMethod(CLMethod):
    name = "coda"

    def __init__(self, model, config):
        super().__init__(model, config)
        # 从主干推导宽度，不硬编码 768（换非 ViT-B/16 主干时会静默出错）
        dim = model.encoder.pos_embedding.shape[-1]
        n = config.prompt_pool_size
        length = config.prompt_length
        self.selector = CODASelector(n, length, dim)
        model.encoder = PromptedEncoder(model.encoder, self.selector, length)

    def regularization_loss(self):
        return 0.0
