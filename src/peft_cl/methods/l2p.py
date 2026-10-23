"""L2P 基线（Learning to Prompt，Wang et al. CVPR 2022）。

核心：冻结 ViT，维护一个 prompt 池（M 个 prompt，每个长 L），每输入用 query-key
匹配选 top-k 个 prompt 拼到序列前。prompt 池在任务间共享，靠「不同任务激活不同
prompt」减轻遗忘。这里复现其 prompt 池 + 余弦 top-k 选择 + 冻结主干，属忠实简化版。

query 用 patches 的 mean pooling（冻结特征），keys 可学习。top-k 是硬选择、对 keys
不可微，故按原论文加「pull constraint」key loss（把选中的 key 拉向 query），经
regularization_loss 加到总损失上（用上一 batch 的 query，1 步滞后，基线可接受）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapters.prompt import PromptedEncoder
from .base import CLMethod

# key loss 系数：把选中的 key 拉向 query 的力度（太小则 keys 学不动，太大则挤压 CE）
PULL_COEFF = 0.1


class L2PSelector(nn.Module):
    def __init__(self, pool_size: int, prompt_len: int, topk: int, dim: int):
        super().__init__()
        self.prompt_len = prompt_len
        self.topk = topk
        # 小方差初始化，避免 prompt 一开始就扰乱冻结特征
        self.prompt_pool = nn.Parameter(torch.randn(pool_size, prompt_len, dim) * 0.02)
        self.keys = nn.Parameter(torch.randn(pool_size, dim) * 0.02)
        self._last_q = None
        self._last_idx = None

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        q = F.normalize(patches.mean(dim=1), dim=-1)          # (B, D)
        k = F.normalize(self.keys, dim=-1)                    # (M, D)
        sim = q @ k.t()                                       # (B, M)
        idx = sim.topk(self.topk, dim=-1).indices             # (B, topk)
        self._last_q = q.detach()                             # 存下来供 key_loss 用
        self._last_idx = idx
        prompt = self.prompt_pool[idx]                        # (B, topk, L, D)
        return prompt.reshape(prompt.shape[0], -1, prompt.shape[-1])  # (B, topk*L, D)

    def key_loss(self) -> torch.Tensor:
        """把选中的 key 拉向对应 query（最大化余弦相似度），训练 keys。"""
        if self._last_q is None or self._last_idx is None:
            return torch.zeros((), device=self.keys.device)
        q = self._last_q.to(self.keys.device)                 # 已 normalize
        k = F.normalize(self.keys[self._last_idx], dim=-1)    # (B, topk, D)
        cos = (q.unsqueeze(1) * k).sum(-1)                    # (B, topk)
        return (1.0 - cos).mean()


class L2PMethod(CLMethod):
    name = "l2p"

    def __init__(self, model, config):
        super().__init__(model, config)
        # 从主干推导宽度，不硬编码 768（换非 ViT-B/16 主干时会静默出错）
        dim = model.encoder.pos_embedding.shape[-1]
        pool = config.prompt_pool_size
        length = config.prompt_length
        topk = config.prompt_topk
        self.selector = L2PSelector(pool, length, topk, dim)
        model.encoder = PromptedEncoder(model.encoder, self.selector, length * topk)

    def regularization_loss(self):
        return PULL_COEFF * self.selector.key_loss()
