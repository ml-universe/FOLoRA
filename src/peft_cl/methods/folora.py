"""FOLoRA（本文方法）：Fisher 正交化的低秩适配。

机制：所有任务共享同一个低秩 adapter（参数预算固定）。每完成一个任务，用该任务的
数据估计「输出方向二阶核」C_τ = E[g·gᵀ] 并累加进累积核 Ḡ；训练新任务时，把 Ḡ 的
top-k 特征方向（重要性=特征值）作为「保护子空间」，加正则惩罚新更新的投影：

    Ω = λ · Σ_j σ_j ‖u_jᵀ ΔW‖²

重要方向（高特征值）强保护防遗忘，次要方向允许复用以保塑性。详见论文 §4。
"""

import torch

from ..adapters.lora import inject_lora
from ..fisher.fisher import output_kernel_fisher
from .base import CLMethod


class FOLoRAMethod(CLMethod):
    name = "folora"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.loras = inject_lora(model, config.lora_rank, config.lora_alpha)
        self.acc_kernel = {}    # lora 下标 -> 累积核 Ḡ（out×out，存 CPU）
        self.ref_delta = {}     # lora 下标 -> 上一任务结束后的 ΔW 快照（device）
        self.protected = {}     # lora 下标 -> (特征值 topk, 特征向量 topk)，本任务用

    def before_task(self, task_id):
        self.protected = {}
        dev = self.config.device
        # 恢复时 ref_delta 可能还在 CPU，移到当前设备
        self.ref_delta = {k: v.to(dev) for k, v in self.ref_delta.items()}
        if not self.acc_kernel:
            return
        k = self.config.folora_topk or self.config.lora_rank
        for i, G in self.acc_kernel.items():
            G = G.to(dev)
            w, V = torch.linalg.eigh(G)          # 升序，对称矩阵特征分解
            kk = min(k, w.shape[0])
            w = w[-kk:]                          # 取最大 kk 个特征值
            V = V[:, -kk:]                       # 对应特征向量 (out, kk)
            self.protected[i] = (w, V)

    def after_task(self, task_id, train_loader, device):
        kernels = output_kernel_fisher(self.model, train_loader,
                                       self.config.fisher_batches, device)
        for i, C in enumerate(kernels):
            if C is None:
                continue
            if i not in self.acc_kernel:
                self.acc_kernel[i] = C.detach().cpu()
            else:
                self.acc_kernel[i] = self.acc_kernel[i] + C.detach().cpu()
        # 记录本任务结束时的 ΔW 快照，作为下一任务正则的「参考点」。
        # 关键：正则要惩罚的是「相对参考点的变化量 δ = ΔW − ΔW_ref」，不是绝对量 ΔW。
        # 若惩罚绝对量，任务开始即 reg>0，会把老任务已学到的方向（ΔW_ref 与老方向的重叠）
        # 一起往外推，等于主动抹掉老知识（比 EWC 还差）。
        for i, lora in enumerate(self.loras):
            self.ref_delta[i] = lora.delta_weight.detach().clone()

    def regularization_loss(self):
        if not self.protected:
            return 0.0
        loss = 0.0
        for i, (w, V) in self.protected.items():
            dW = self.loras[i].delta_weight - self.ref_delta[i]   # 变化量 (out, in)
            proj = V.t() @ dW                                     # (k, in)
            loss = loss + (w * (proj ** 2).sum(dim=1)).sum()
        return self.config.folora_lambda * loss

    def state_dict(self):
        return {
            "acc_kernel": {str(i): v.cpu() for i, v in self.acc_kernel.items()},
            "ref_delta": {str(i): v.cpu() for i, v in self.ref_delta.items()},
        }

    def load_state_dict(self, d):
        self.acc_kernel = {int(k): v for k, v in d["acc_kernel"].items()}
        self.ref_delta = {int(k): v for k, v in d.get("ref_delta", {}).items()}
