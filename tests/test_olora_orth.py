"""O-LoRA 正交性约束（regularization_loss）测试，纯 CPU。

为什么值得单独测
----------------
这一项**默认关（λ₁=0）**，所以它不会在主队列里被任何一个 run 覆盖到——一旦写错，
只有专门跑去开它的那次实验会得到错的结果，而且不会有任何报错。有两个失效模式：

1. **转置约定搞反**：原文 A∈R^{d×r}、约束 A_iᵀA_t=0；本仓库 `lora_A.weight` 是 (r,d)。
   若照抄原文写成 `A_t.T @ A_i`，形状上会在 r≠d 时直接崩，但若 r==d（小测试里很容易）
   则**静默算成另一个量**（自身行内积 vs 跨矩阵内积），值看着也「像个惩罚」。
   这里用 r≠d 且手工对角化的构造把方向钉死。
2. **历史栈缓存失效时机错**：缓存若不在每个任务开始时失效，第二个任务起用的还是
   上一个任务的历史——损失会偏小地「看起来正常」。这里跨两个任务验证。

测的是不变量：λ₁=0 恒为 0、正交时惩罚为 0、偏离正交时惩罚 > 0、量值与手工一致。
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from peft_cl.adapters.multi_lora import inject_multi_lora
from peft_cl.methods.olora import OLoRAMethod
from peft_cl.utils.config import CLConfig


class Tiny(nn.Module):
    def __init__(self, in_dim=8, n_cls=5):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 6), nn.Linear(6, 6),
                                 nn.Linear(6, 6), nn.Linear(6, n_cls))

    def forward(self, x):
        return self.mlp(x)


def build(rank=2, orth_lambda=0.1):
    torch.manual_seed(0)
    cfg = CLConfig(method="olora", lora_rank=rank, lora_alpha=rank,
                   num_tasks=3, olora_orth_lambda=orth_lambda)
    model = Tiny()
    method = OLoRAMethod(model, cfg)
    return model, method, cfg


def test_zero_lambda_is_exactly_zero():
    """λ₁=0（默认）必须**恒等于** 0.0，而不是「很小的数」。

    这保证已完成的那 10 个 O-LoRA run 的语义逐位不变：默认值下这一项完全不参与训练。
    """
    _, method, _ = build(orth_lambda=0.0)
    method.before_task(0)
    assert method.regularization_loss() == 0.0
    # 即使已经有历史 A，λ₁=0 也必须严格 0（不能因为算了个非零量再乘 0 而留下浮点残渣）
    method.after_task(0, None, None)
    method.before_task(1)
    out = method.regularization_loss()
    assert out == 0.0, f"λ₁=0 时应严格为 0.0，实际 {out!r}"


def test_penalty_matches_bruteforce_and_convention():
    """惩罚 = λ₁·Σ_{i<t} ‖ours_A_t @ ours_A_iᵀ‖_F²，且用 r≠d 钉死转置方向。"""
    rank, lam = 2, 0.35
    _, method, _ = build(rank=rank, orth_lambda=lam)
    method.before_task(0)

    # 造一个「历史 A」：手工塞成标准基，便于心算
    hist = torch.zeros(rank, method.multi_loras[0].in_dim)
    hist[0, 0] = 1.0
    hist[1, 1] = 1.0
    for i in range(len(method.multi_loras)):
        method.prev_A[i] = [hist.clone()]
    method._cache = {}          # 模拟 before_task 的失效

    # 当前 A：第 0 行与 hist 第 0 行内积 = 1（不正交），第 1 行与 hist 正交
    cur = torch.zeros(rank, method.multi_loras[0].in_dim)
    cur[0, 0] = 1.0
    cur[1, 5] = 1.0
    for i, ml in enumerate(method.multi_loras):
        with torch.no_grad():
            ml.adapters[-1].lora_A.weight.copy_(cur)

    # 手工：A_t @ A_histᵀ = [[1,0],[0,0]] -> ‖·‖_F² = 1，每层 1，乘层数乘 λ
    expect = lam * 1.0 * len(method.multi_loras)
    got = float(method.regularization_loss().detach())
    assert got == pytest.approx(expect, rel=1e-6), (
        f"正交性惩罚与手工值不符：got={got} expect={expect}"
        "（转置方向错了就会得到另一个量）")

    # 若把第 1 行也变成与 hist 第 1 行重合，惩罚应再增加 1（每层）
    with torch.no_grad():
        for ml in method.multi_loras:
            ml.adapters[-1].lora_A.weight[1, 5] = 0.0
            ml.adapters[-1].lora_A.weight[1, 1] = 1.0
    expect2 = lam * 2.0 * len(method.multi_loras)
    assert float(method.regularization_loss().detach()) == pytest.approx(expect2, rel=1e-6)


def test_orthogonal_current_has_zero_penalty():
    """当前 A 与历史 A 完全正交时，惩罚必须精确为 0（不是「接近 0」的松弛项）。"""
    _, method, _ = build(rank=2, orth_lambda=1.0)
    method.before_task(0)
    hist = torch.zeros(2, method.multi_loras[0].in_dim)
    hist[0, 0] = 1.0
    hist[1, 1] = 1.0
    for i in range(len(method.multi_loras)):
        method.prev_A[i] = [hist.clone()]
    method._cache = {}
    with torch.no_grad():
        for ml in method.multi_loras:
            w = ml.adapters[-1].lora_A.weight
            w.zero_()
            w[0, 4] = 1.0          # 与 hist 的两行都正交（hist 是 e_0, e_1）
            w[1, 5] = 1.0
    assert float(method.regularization_loss().detach()) == pytest.approx(0.0, abs=1e-12)


def test_cache_invalidated_between_tasks():
    """第二个任务的历史栈必须包含第一个任务的 A（缓存失效时机）。

    若缓存不在 before_task 失效，task 1 仍用「task 0 训练前」的空/旧历史，惩罚会偏小
    且不报错——属于最隐蔽的一类。
    """
    _, method, _ = build(rank=2, orth_lambda=1.0)
    method.before_task(0)
    # 任务 0 结束前把它的 A 设成已知值，after_task 会把它并入 prev_A
    with torch.no_grad():
        for ml in method.multi_loras:
            w = ml.adapters[-1].lora_A.weight
            w.zero_()
            w[0, 0] = 1.0
    method.after_task(0, None, None)

    method.before_task(1)          # 应触发缓存失效
    with torch.no_grad():
        for ml in method.multi_loras:
            w = ml.adapters[-1].lora_A.weight
            w.zero_()
            w[0, 0] = 1.0          # 与任务 0 的 A 完全重合 -> 每层惩罚 = 1
    expect = 1.0 * 1.0 * len(method.multi_loras)
    assert float(method.regularization_loss().detach()) == pytest.approx(expect, rel=1e-6), (
        "task 1 的惩罚没有把 task 0 的历史算进去：缓存失效时机错了")


def test_loss_is_differentiable_wrt_current_A_only():
    """惩罚必须对**当前** A 有梯度、对历史 A 无梯度（历史已 detach）。

    历史若带梯度，会把梯度回传到已冻结的旧 adapter 上——不仅白算，还会让
    优化器更新那些参数不该更新的历史参数（requires_grad=False 时 optimizer 不会更新，
    但梯度图会白白占显存）。
    """
    _, method, _ = build(rank=2, orth_lambda=1.0)
    method.before_task(0)
    hist = torch.zeros(2, method.multi_loras[0].in_dim)
    hist[0, 0] = 1.0
    for i in range(len(method.multi_loras)):
        method.prev_A[i] = [hist.clone()]
    method._cache = {}

    loss = method.regularization_loss()
    assert torch.is_tensor(loss) and loss.requires_grad
    loss.backward()
    for ml in method.multi_loras:
        assert ml.adapters[-1].lora_A.weight.grad is not None
        assert float(ml.adapters[-1].lora_A.weight.grad.abs().sum()) > 0
    # 历史栈本身不能挂 grad
    for k, v in method._cache.items():
        assert not v.requires_grad, "历史 A 缓存带梯度，应 detach"


def test_first_task_has_no_history_and_no_penalty():
    """第一个任务没有历史任务，惩罚必须为 0（不能因为空列表而崩或算 NaN）。"""
    _, method, _ = build(rank=2, orth_lambda=1.0)
    method.before_task(0)
    out = method.regularization_loss()
    assert float(out) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(float(out))
