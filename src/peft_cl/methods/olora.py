"""O-LoRA 基线（复现其核心思想）：每个任务分配一个 LoRA，新任务的 A 正交初始化。

O-LoRA（Wang et al., Findings of EMNLP 2023）的核心：把新任务的 A 矩阵初始化到
「与历史任务的 A 行空间正交」的子空间上，从而让不同任务占据近似不重叠的低秩子空间、
减少干扰。这里复现其正交初始化 + 逐任务独立 adapter，属忠实简化版。

**推理聚合方式**（`config.olora_aggregate`）——注意这不是「复现出错」，而正是
O-LoRA 原文的设计：其原文明确声明「推理阶段不需要 task id」（task-ID-free
inference），因此**必须把全部任务的增量合并**送给同一个前向，不能用 task id 选。
  - "sum"（默认）：ΔW = Σ_i B_i A_i，即论文原式。在 20 任务 × rank 16 的设定下，
    ΔW 的秩上限是 320/768 ≈ 42%（不再「低秩」），实测 NCM 64.79 ± 2.08 (n=10)，
    低于 SimpleCIL 冻结特征地板 70.31、也低于 Seq-LoRA 69.31。
  - "mean"：ΔW = (1/T) Σ_i B_i A_i，**诊断用**的缩放对照——用来区分「合并的秩增长」
    与「合并的幅度缩放」哪个是主因。两种都报。
  - TIL（给 task id、走单个 adapter）82.8%，说明正交子空间本身是学到的、
    失效发生在**任务无关的合并推理**这一步，而非训练。
"""

import torch

from ..adapters.multi_lora import inject_multi_lora
from .base import CLMethod


class OLoRAMethod(CLMethod):
    name = "olora"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.multi_loras = inject_multi_lora(
            model, config.lora_rank, config.lora_alpha,
            aggregate=getattr(config, "olora_aggregate", "sum"))
        self.prev_A = {i: [] for i in range(len(self.multi_loras))}  # 每层历史 A 列表
        # 历史 A 的拼接缓存（避免每个 step 重复 cat + H2D）；每个任务开始时失效重建
        self._cache = {}

    def before_task(self, task_id, train_loader=None):
        dev = self.config.device
        # 历史 A 在本任务开始前已经定型（after_task 追加过上一任务），缓存此处失效重建。
        # 放在 __init__ 与 before_task 里而非 after_task：after_task 之后 prev_A 才变，
        # 而下一任务的第一次 regularization_loss 调用会按需重建。
        self._cache = {}
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
        """O-LoRA 的正交性约束 λ₁·Σ_{i<t} ‖A_tᵀ A_i‖_F²。

        **为什么必须有这一项**：O-LoRA 原文的正交性不是「初始化」而是**训练期约束**——
        目标函数是 Σ log p(y|x) + λ₁ Σ_{i<t} ‖A_tᵀ A_i‖_F²。只做正交初始化、训练时不加
        惩罚，第一个梯度步就会让 A_t 离开正交互补空间，于是我们测的其实是「O-LoRA 的
        初始化 + 无约束微调」，比原方法**更弱**。基线被做弱是审稿人最容易抓的点（和
        EWC 的 λ 调小是同一类问题），所以这一项要能开、并且把 λ₁ 一起报出来。

        **记号约定**：原文 A ∈ R^{d×r}，子空间 U_t = span{A_t 的**列**}，约束 A_iᵀA_t = 0。
        本仓库的 `lora_A.weight` 形状是 (r, d) = 原文 Aᵀ，故原文的「列」= 本仓库的「行」，
        于是 A_iᵀ A_t 的 (j,k) 元 = ⟨本仓库 A_i 的第 j 行, 本仓库 A_t 的第 k 行⟩，
        即 `ours_A_i @ ours_A_t.T`。Frobenius 范数在转置下不变，所以
        ‖A_tᵀ A_i‖_F² = ‖ours_A_t @ ours_A_iᵀ‖_F²，下面按后者实现。

        λ₁ = 0.0（默认）时直接返回 0.0 —— 与已完成的那批 run 语义一致。
        """
        lam = getattr(self.config, "olora_orth_lambda", 0.0)
        if not lam:
            return 0.0
        loss = 0.0
        for i, ml in enumerate(self.multi_loras):
            hist = self.prev_A.get(i)
            if not hist or not ml.adapters:
                continue
            A_t = ml.adapters[-1].lora_A.weight        # (r, in)，当前任务，带 grad
            A_hist = self._hist_stack(i)               # (t·r, in)，历史任务，已 detach
            if A_hist is None or A_hist.shape[0] == 0:
                continue
            # (r, t·r)：一次 matmul 拿到全部内积，等价于对每个历史任务单独算再求和
            loss = loss + (A_t @ A_hist.t()).pow(2).sum()
        return lam * loss

    def _hist_stack(self, i):
        """把第 i 层的历史 A 缓存在 device 上，避免每个 step 重复 cat + H2D。

        在 before_task 里构建（那时 prev_A 恰好是 A_0..A_{t-1}）；断点续训时
        prev_A 由 load_state_dict 恢复，随后的 before_task 会重建缓存。
        """
        key = ("stack", i)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        hist = self.prev_A.get(i)
        if not hist or not self.multi_loras:
            return None
        dev = self.multi_loras[i].adapters[-1].lora_A.weight.device \
            if self.multi_loras[i].adapters else "cpu"
        stack = torch.cat([a.to(dev) for a in hist], dim=0).detach()
        self._cache[key] = stack
        return stack

    def rebuild_for_resume(self, last_task_id):
        """补建 last_task_id+1 个 per-task adapter，使 checkpoint 的键能对上。"""
        for ml in self.multi_loras:
            for _ in range(last_task_id + 1):
                ml.add_task()

    def state_dict(self):
        return {"prev_A": {str(i): [t.cpu() for t in v] for i, v in self.prev_A.items()}}

    def load_state_dict(self, d):
        self.prev_A = {int(k): list(v) for k, v in d["prev_A"].items()}
        self._cache = {}          # prev_A 被整体替换，缓存必须失效
