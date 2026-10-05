"""InfLoRA 基线（Liang & Li, CVPR 2024, *Interference-Free Low-Rank Adaptation*）。

原论文要点（复现口径见下）
--------------------------
冻结预训练权重 W，为每个任务 t 追加一个低秩分支：
    e = W_{t-1} h + A_t B_t h ,   W_t = W_{t-1} + A_t B_t
其中 **B_t ∈ R^{r×d_in} 是降维矩阵（对应 LoRA 的 A）**，
     **A_t ∈ R^{d_out×r} 是升维矩阵（对应 LoRA 的 B）**。

- 学习任务 t 之前**先设计好 B_t 并冻结**，A_t 初始化为 0 且是唯一可训练参数；
  旧分支与 W 全程冻结。A_t=0 保证新分支起步时是恒等（不扰动已有解）。
- 设计准则：B_t 的行空间必须落在  N_t ∩ M_t^⊥
    * N_t  —— 新任务的梯度空间。对线性层，梯度形如 G·H_tᵀ，故落在输入矩阵 H_t
              的行空间里，用 H_t 的 top-r 左奇异向量近似；
    * M_t  —— 旧任务的梯度空间（即旧任务输入空间的并），M_t^⊥ 是其正交补。
              「ΔW 对旧任务无干扰」等价于 ΔW·h = 0, ∀h ∈ span(H_old)，
              即 B_t 的行空间 ⊥ span(H_old)。
  两条合起来 = 「新分支只在新任务需要、且旧任务不在乎的方向上动」，
  这正是原论文 stability（正交于旧梯度）与 plasticity（落在新梯度空间）的权衡。

为什么对应到本仓库的 _Adapter：lora_A 是降维 (r×in)、lora_B 是升维 (out×r)，
正是原论文的 B_t / A_t。所以本实现把 lora_A 设为设计好的正交基并冻结，
lora_B 初始化为 0 并训练。

复现口径（诚实声明，需写进论文）
--------------------------------
原论文把模块插入 ViT 各 block 的 attention K/V，主干为 ImageNet-21k 预训练的
ViT-B/16。本仓库的统一实验架构是「LoRA 插在 mlp.3（MLP 下投影）」，为使比较
公平（所有方法共用同一注入点与预算），这里把 InfLoRA 的子空间构造原样搬到了
mlp.3 上。顶层超参 r 与原论文一致（默认 16）。
"""

import torch

from ..adapters.multi_lora import inject_multi_lora
from .base import CLMethod


def _orth_basis(M: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """返回 M 列空间的一组标准正交基（M 为 (d, m)），按奇异值降序保留非退化列。"""
    if M.shape[1] == 0:
        return M
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    keep = S > tol * max(1.0, float(S[0])) if S.numel() else torch.zeros(0, dtype=torch.bool)
    return U[:, keep]


def _input_basis(rows: torch.Tensor, r: int) -> torch.Tensor:
    """由输入样本矩阵 rows (n, d_in) 估计输入空间 top-r 方向，返回 (d_in, <=r)，标准正交。

    输入样本张成的空间 = rows 的**行空间** = 右奇异向量张成的空间，故取 V 而非 U
    （等价于 HᵀH 的前 r 个特征向量）。用 svd_lowrank 只算前 q 个三元组，
    避免对 (n, d_in) 做全 SVD —— n 可达数万。
    """
    n, d = rows.shape
    q = min(r, n, d)
    _, _, V = torch.svd_lowrank(rows, q=q)      # 注意：在 torch.svd_lowrank，不在 torch.linalg
    return _orth_basis(V.float())


class InfLoRAMethod(CLMethod):
    """InfLoRA：逐任务追加分支，新分支的降维矩阵被设计成 N_t ∩ M_t^⊥。"""

    name = "inflora"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.multi_loras = inject_multi_lora(model, config.lora_rank, config.lora_alpha)
        # 每层累积的旧任务输入空间基（d_in × m，CPU 上保存以便序列化）
        self.old_basis = {i: None for i in range(len(self.multi_loras))}
        # 每层逐任务设计出的降维基（用于断点续训恢复：冻结参数不进 model_state）
        self.designed_A = {i: [] for i in range(len(self.multi_loras))}

    # ---------- 输入空间收集 ----------

    @torch.no_grad()
    def _collect_inputs(self, train_loader, device, max_rows: int = 40000):
        """对每个注入层收集新任务的输入样本，返回 list[(n, d_in) fp32 CPU 张量]。"""
        model = self.model
        was_training = model.training
        model.eval()
        buckets = [[] for _ in self.multi_loras]
        handles = []
        for i, ml in enumerate(self.multi_loras):
            def hook(module, inp, out, i=i):
                x = inp[0].detach()
                buckets[i].append(x.reshape(-1, x.shape[-1]).float().cpu())
            handles.append(ml.base.register_forward_hook(hook))
        def collected() -> int:
            return sum(b.numel() // b.shape[-1] for b in buckets[0]) if buckets[0] else 0

        try:
            for x, _ in train_loader:
                model(x.to(device))
                if collected() >= max_rows:
                    break
        finally:
            for h in handles:
                h.remove()
            model.train(was_training)
        return [torch.cat(b, dim=0)[:max_rows] if b else None for b in buckets]

    # ---------- 任务钩子 ----------

    def before_task(self, task_id, train_loader=None):
        dev = self.config.device
        r = self.config.lora_rank

        if train_loader is None:
            raise ValueError("InfLoRA 需要 train_loader 来设计新任务的子空间")

        # 1) 先收集「当前模型」下新任务的输入空间（必须在加新分支之前做，
        #    否则新分支会改变 forward 的输入分布）
        rows_per_layer = self._collect_inputs(train_loader, dev)

        for i, ml in enumerate(self.multi_loras):
            a = ml.add_task()
            # 旧分支全部冻结
            for prev in ml.adapters[:-1]:
                for p in prev.parameters():
                    p.requires_grad = False
            # 新分支：降维矩阵冻结且被设计，升维矩阵可训练且初始化为 0
            for p in a.lora_A.parameters():
                p.requires_grad = False
            for p in a.lora_B.parameters():
                p.requires_grad = True
            a.lora_B.weight.data.zero_()

            rows = rows_per_layer[i]
            d_in = a.lora_A.weight.shape[1]
            if rows is None or rows.shape[0] < r:
                # 极端退化情形：没有可用输入，退化为随机正交基（等价于 LoRA 随机初始化）
                basis = _orth_basis(torch.randn(d_in, r, device=dev))
            else:
                basis = _input_basis(rows.to(dev), r)          # (d_in, <=r)，标准正交

                # 2) 投影掉旧任务输入空间：N_t ∩ M_t^⊥
                M = self.old_basis[i]
                if M is not None and M.shape[1] > 0:
                    M = M.to(dev)
                    basis = basis - M @ (M.t() @ basis)
                    basis = _orth_basis(basis)                  # 重新标准正交化

                # 3) 把**新任务的完整输入空间**并入旧空间（累积保护范围）。
                #    并入用的是**投影后**的 `basis`（即上面第 2 步的结果，不另存一份）。
                #    这与「并入投影前的输入空间」等价：投影只去掉了旧空间分量，而旧空间
                #    分量本就在 M 中，故 span(M, basis_投影后) = span(M, basis_投影前)。
                #    （原注释此处还有半句「故在投影前记录」，与紧邻的代码相反——代码取的
                #    是投影后的 basis。2026-10-05 删除该半句，它描述的是没被采纳的写法。）
                M_old = self.old_basis[i]
                M_new = basis if M_old is None else torch.cat([M_old.to(dev), basis], dim=1)
                self.old_basis[i] = _orth_basis(M_new).cpu()

                if basis.shape[1] < r:                          # 秩不足则补齐
                    basis = self._pad_basis(basis, r, d_in, dev, M)
            basis = basis[:, :r]                                # 保证恰好 r 列

            with torch.no_grad():
                a.lora_A.weight.data = basis.t().to(a.lora_A.weight.device).to(
                    a.lora_A.weight.dtype)
            self.designed_A[i].append(a.lora_A.weight.detach().cpu().clone())

    @staticmethod
    def _pad_basis(basis: torch.Tensor, r: int, d_in: int, dev,
                   M: torch.Tensor = None) -> torch.Tensor:
        """当投影后秩不足 r 时，用随机方向补齐到 r（保持标准正交，且仍 ⊥ M）。"""
        need = r - basis.shape[1]
        rand = torch.randn(d_in, need + 8, device=dev)
        if basis.shape[1] > 0:
            rand = rand - basis @ (basis.t() @ rand)
        if M is not None and M.shape[1] > 0:
            rand = rand - M @ (M.t() @ rand)
        extra = _orth_basis(rand)
        return torch.cat([basis, extra[:, :need]], dim=1)

    def after_task(self, task_id, train_loader, device):
        pass  # 输入空间已在 before_task 中累积

    def regularization_loss(self):
        return 0.0

    def rebuild_for_resume(self, last_task_id):
        """补建 last_task_id+1 个 per-task 分支，使 checkpoint 的键能对上。

        设计出来的降维矩阵 lora_A 是冻结参数（requires_grad=False），不在
        checkpoint 的 model_state 里，由下面的 load_state_dict 从 method_state 恢复。
        """
        for ml in self.multi_loras:
            for _ in range(last_task_id + 1):
                ml.add_task()

    def state_dict(self):
        return {
            "old_basis": {str(i): (v.cpu() if v is not None else None)
                          for i, v in self.old_basis.items()},
            # 设计出的降维基必须单独存：它是冻结参数，不会进 model_state
            "designed_A": {str(i): [a.cpu() for a in v]
                           for i, v in self.designed_A.items()},
        }

    def load_state_dict(self, d):
        self.old_basis = {int(k): v for k, v in d["old_basis"].items()}
        self.designed_A = {int(k): list(v) for k, v in (d.get("designed_A") or {}).items()}
        for i, mats in self.designed_A.items():
            adapters = self.multi_loras[i].adapters
            for j, A in enumerate(mats):
                if j < len(adapters):
                    with torch.no_grad():
                        adapters[j].lora_A.weight.copy_(
                            A.to(adapters[j].lora_A.weight.device).to(
                                adapters[j].lora_A.weight.dtype))
