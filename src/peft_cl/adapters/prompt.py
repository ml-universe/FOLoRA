"""Prompt 注入：在冻结 ViT 的 CLS token 之后、patch 之前插入可学习 prompt。

torchvision ViT 的 forward 顺序：conv_proj 得到 patch embedding → 拼接 class_token
→ encoder（+pos_embedding → layers → ln）→ 取 [:, 0] 的 CLS → head。
prompt 方法（L2P / CODA）在 encoder 内部、加 pos_embedding 之前，把 prompt 插到
[CLS, prompt, patches] 的中间，CLS 仍保持在位置 0（head 取 x[:, 0] 不受影响）。

prompt 用「共享可学习位置编码」prompt_pos（所有 prompt token 复用同一个位置），
与原 pos_embedding（CLS + 196 patch）拼接。
"""

import torch
import torch.nn as nn


class PromptedEncoder(nn.Module):
    """包装 torchvision ViT Encoder，前向在 CLS 后插入 prompt。

    selector：可学习的 prompt 选择器（L2P 的 top-k / CODA 的注意力加权），
    输入 patches (B, N_patch, D)，输出 prompt (B, L_total, D)。
    """

    def __init__(self, encoder: nn.Module, selector: nn.Module, prompt_total_len: int):
        super().__init__()
        self.layers = encoder.layers           # 复用冻结的 transformer 层
        self.ln = encoder.ln                   # 复用冻结的 layer norm
        self.pos_embedding = encoder.pos_embedding  # 冻结 (1, 197, D)
        self.selector = selector
        D = encoder.pos_embedding.shape[-1]
        self.prompt_pos = nn.Parameter(torch.zeros(1, prompt_total_len, D))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input = [CLS, patches]，shape (B, 197, D)
        cls = input[:, :1]
        patches = input[:, 1:]
        prompt = self.selector(patches)                     # (B, L_total, D)
        x = torch.cat([cls, prompt, patches], dim=1)        # (B, 1+L_total+196, D)
        pos = torch.cat([
            self.pos_embedding[:, :1],
            self.prompt_pos,
            self.pos_embedding[:, 1:],
        ], dim=1)                                           # (1, 1+L_total+196, D)
        x = x + pos
        x = self.layers(x)
        x = self.ln(x)
        return x
