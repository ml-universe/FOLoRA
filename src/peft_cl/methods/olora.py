"""O-LoRA 基线（复现其核心思想）：每个任务分配一个 LoRA，新任务的 A 正交初始化。

O-LoRA（Wang et al., ICLR 2024）的核心：把新任务的 A 矩阵初始化到「与历史任务的
A 行空间正交」的子空间上，从而让不同任务占据近似不重叠的低秩子空间、减少干扰。
这里复现其正交初始化 + 逐任务独立 adapter（推理时累加所有 adapter），属忠实简化版。
"""

import torch

from ..adapters.multi_lora import inject_multi_lora
from .base import CLMethod


class OLoRAMethod(CLMethod):
    name = "olora"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.multi_loras = inject_multi_lora(model, config.lora_rank, config.lora_alpha)
        self.prev_A = {i: [] for i in range(len(self.multi_loras))}  # 每层历史 A 列表

    def before_task(self, task_id):
        dev = self.config.device
        for i, ml in enumerate(self.multi_loras):
            a = ml.add_task()
            # 只训练当前任务的 adapter，冻结历史 adapter
            for prev in ml.adapters[:-1]:
                for p in prev.parameters():
                    p.requires_grad = False
            for p in ml.adapters[-1].parameters():
                p.requires_grad = True
            # 正交初始化：A_new ← A_new (I - UᵀU)，U 为历史 A 行空间正交基
            if self.prev_A[i]:
                A_stack = torch.cat(self.prev_A[i], dim=0).to(dev)  # (t·r, in)
                _, _, Vh = torch.linalg.svd(A_stack, full_matrices=False)
                U_row = Vh.to(a.lora_A.weight.device)               # (k, in) 行正交
                proj = U_row.t() @ U_row                            # (in, in) 投影阵
                with torch.no_grad():
                    a.lora_A.weight.data = a.lora_A.weight.data - a.lora_A.weight.data @ proj

    def after_task(self, task_id, train_loader, device):
        # 把训练后的 A 并入历史子空间
        for i, ml in enumerate(self.multi_loras):
            A = ml.adapters[-1].lora_A.weight.detach().cpu()
            self.prev_A[i].append(A)

    def regularization_loss(self):
        return 0.0

    def state_dict(self):
        return {"prev_A": {str(i): [t.cpu() for t in v] for i, v in self.prev_A.items()}}

    def load_state_dict(self, d):
        self.prev_A = {int(k): list(v) for k, v in d["prev_A"].items()}
