"""FOLoRA v2：参数空间 Fisher 低秩投影（修复 v1 输出空间正则失效）。

v1 的问题：正则惩罚「输出方向投影」u_jᵀ δW，u_j 是输出 Fisher 核 E[g_δ g_δᵀ] 的
top 方向。但理论遗忘上界是 δθᵀ F_θ δθ，F_θ 是「参数梯度」的 Fisher E[g_θ g_θᵀ]，
两者差一个输入 x（链式法则 g_θᵀ δθ = g_δᵀ δW x）。v1 把 x 积分掉了，保护方向与
真实遗忘方向不对齐，实测正则对训练几乎 no-op（遗忘 ≈ seq）。

v2：直接估「参数梯度」g_θ = (g_A, g_B) 的逐样本外积，维护累积 Fisher 的低秩
近似 Ĝ（top-k 主方向，按奇异值加权做增量 PCA），正则 Ω = λ ‖Ĝ δθ‖²，
δθ = θ − θ_ref（θ 为展平的 A、B，θ_ref 为上一任务快照）。这正好最小化理论的
δθᵀ F_θ δθ 上界，且比 EWC 的对角 Fisher 多了参数相关性（低秩全协方差）——正是
论文「Fisher 加权低秩投影」卖点的正确落地。
"""

import torch

from ..adapters.lora import inject_lora
from ..fisher.fisher import param_gradients
from .base import CLMethod


def _topk_directions(G: torch.Tensor, k: int) -> torch.Tensor:
    """G (m×d) 的 top-k 主方向（按奇异值加权，等价于低秩近似 G ≈ Ũ Σ Vᵀ）。

    返回 (kk, d)，每行 σ_j v_jᵀ（含能量权重）。用 gram 矩阵 G Gᵀ (m×m) 的谱分解求
    top-k（m << d 时高效），保留 σ_j 权重使后续可与新任务原始梯度拼接做增量 PCA。
    """
    gram = G @ G.t()  # (m, m)
    _, eigvecs = torch.linalg.eigh(gram)  # 升序
    kk = min(k, gram.shape[0])
    return eigvecs[:, -kk:].t() @ G  # (kk, d)


class FOLoRAv2Method(CLMethod):
    name = "folora_v2"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.loras = inject_lora(model, config.lora_rank, config.lora_alpha)
        self.acc_grads = {}   # lora idx -> (k, d) 累积参数 Fisher 低秩方向（CPU）
        self.ref_params = {}  # lora idx -> (d,) 上一任务参数字典快照（device）

    def before_task(self, task_id, train_loader=None):
        dev = self.config.device
        self.ref_params = {k: v.to(dev) for k, v in self.ref_params.items()}

    def after_task(self, task_id, train_loader, device):
        Gs = param_gradients(self.model, train_loader,
                             self.config.fisher_batches, device)
        k = self.config.folora_topk or self.config.lora_rank
        for i, G in enumerate(Gs):
            if G is None:
                continue
            if i in self.acc_grads:
                # acc_grads 存在 CPU（state_dict 序列化），cat 前移到 G 所在设备
                G = torch.cat([self.acc_grads[i].to(G.device), G], dim=0)  # (k_old + n, d)
            self.acc_grads[i] = _topk_directions(G, k).cpu()
        # 记录本任务结束时的参数快照，作为下一任务正则的参考点
        for i, lora in enumerate(self.loras):
            self.ref_params[i] = self._flatten(lora).detach().clone()

    def regularization_loss(self):
        if not self.acc_grads:
            return 0.0
        loss = 0.0
        for i, lora in enumerate(self.loras):
            if i not in self.acc_grads:
                continue
            theta = self._flatten(lora)                 # (d,)，可导（含 grad_fn）
            dtheta = theta - self.ref_params[i]         # (d,)，ref 已 detach
            G = self.acc_grads[i].to(dtheta.device)
            if not getattr(self.config, "folora_weighted", True):
                G = self._equalize_row_weights(G)
            loss = loss + (G @ dtheta).pow(2).sum()
        return self.config.folora_lambda * loss

    @staticmethod
    def _equalize_row_weights(G: torch.Tensor) -> torch.Tensor:
        """等权消融：保持方向不变，把各方向的权重统一成平均奇异值 σ̄。

        G 的每行是 σ_j v_jᵀ，于是 ‖G δθ‖² = Σ_j σ_j² (v_jᵀ δθ)²——正则惩罚天然按
        Fisher 特征值 σ_j² 加权，这正是论文「重要性加权」卖点。等权对照要把这个
        加权去掉，但**不能简单地把行归一化**：那样会连惩罚的整体尺度一起改掉，
        原来的 λ 就不再适用，必须重新扫 λ，而「重扫过的等权 vs 没重扫的加权」
        比出来的差异分不清是加权带来的还是调参带来的。

        这里改成「方向保持、权重齐次化」——每行缩放到共同的 σ̄_rms（保留符号），
        于是两边可以在同一个 λ 下直接比较，消融干净。

        **2026-09-25 修正（此前用算术平均 σ̄，尺度没对齐）**：
        加权版总权重 = Σ_j σ_j²；齐次化到公共模长 m 后总权重 = k·m²。
        要让两者**精确相等**，须取 m = sqrt(mean(σ_j²)) = **RMS**，此时
        k·RMS² = k·mean(σ²) = Σσ_j²，恒等成立（与 σ 的分布无关）。
        曾用 m = mean(σ_j)，那只是**上界**（k·σ̄² ≤ Σσ²，仅当所有 σ_j 相等时取等），
        在真实 σ 分布（1.94–19.03）下实测比值仅 **0.248**——即等权臂的惩罚只有
        加权臂的 1/4，两臂同时差了「分配方式」和「总尺度」两件事，
        效应无法干净归因。改用 RMS 后该混杂消除。
        """
        norms = G.norm(dim=1, keepdim=True).clamp_min(1e-12)   # (k,1) 即 σ_j
        return G / norms * norms.pow(2).mean().sqrt()

    @staticmethod
    def _flatten(lora) -> torch.Tensor:
        return torch.cat([lora.lora_A.weight.reshape(-1),
                          lora.lora_B.weight.reshape(-1)])

    def state_dict(self):
        return {
            "acc_grads": {str(i): v.cpu() for i, v in self.acc_grads.items()},
            "ref_params": {str(i): v.cpu() for i, v in self.ref_params.items()},
        }

    def load_state_dict(self, d):
        self.acc_grads = {int(k): v for k, v in d["acc_grads"].items()}
        self.ref_params = {int(k): v for k, v in d["ref_params"].items()}
