# -*- coding: utf-8 -*-
"""严格序贯原型（E5，`--ncm_strict_proto`）的端到端逻辑测试。

为什么必须在**开跑前**测：这两个函数只在 `trainer._train_task` / `_finalize` 的
`try/except` 里被调用，异常会被记成一条 warning 然后**静默跳过**。也就是说，哪怕
`task_prototypes` 从头就错，9 个 run 也会照常跑完、照常写出 results.json，只是里面
**没有** `strict_acc_cil`。等到 30 小时后去读结果才发现，损失的是整批算力。

所以这里用一个假的 ViT（只暴露 `_process_input` / `class_token` / `encoder` 三个被
`_extract` 用到的接口）和一个假的 split，把两条路径真正跑一遍，并**核对数值语义**：
原型来自 θ_j（存盘时的模型），评估用 θ_T（终态的模型）。

关键语义（错了不会报错，只会给出一个看似合理的数）：
  * `task_prototypes` 只能读**当前任务**的训练集，且标签是全局索引 `t*C .. t*C+C-1`；
  * `strict_final_eval` 必须读**存盘的原型**，不许用终态模型重算——否则测出来的
    就不是「评估期访问历史数据的代价」而是 0。
下面用「终态模型把旧类特征整体挪走」来构造：若实现正确，CIL 会明显掉；
若它偷偷重算原型，CIL 会保持很高。
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from peft_cl.utils.ncm_trace import strict_final_eval, task_prototypes

D = 8          # 特征维度
C = 2          # 每任务类数
T = 3          # 任务数


class _FakeViT(nn.Module):
    """只实现 `_extract` 用到的那三个接口。

    `_process_input` 把 (B, D) 输入直接穿过；`encoder` 对 [CLS] 与 patch 的拼接结果
    取均值再线性映射——形状无关紧要，只要「同输入同输出、且可被 state 改变」即可。
    """

    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, D))
        self.lin = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(D))
        self.shift = shift

    def _process_input(self, x):
        return x.unsqueeze(1) + self.shift      # (B, 1, D)

    def encoder(self, z):
        return z.mean(dim=1, keepdim=True)      # (B, 1, D)


class _FakeSplit:
    """`task_dataset(task_id, split)` 返回该任务的 TensorDataset，标签为全局索引。"""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self._data = {}
        for t in range(T):
            # 每个类的中心互相分开，保证 NCM 能分对
            centers = torch.randn(C, D, generator=g) * 4.0
            xs, ys = [], []
            for i in range(C):
                xs.append(centers[i] + 0.1 * torch.randn(16, D, generator=g))
                ys.append(torch.full((16,), t * C + i, dtype=torch.long))
            self._data[t] = (torch.cat(xs), torch.cat(ys))

    def task_dataset(self, task_id, split):
        x, y = self._data[task_id]
        return torch.utils.data.TensorDataset(x, y)


def _proto_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_task_prototypes_shape_and_normalisation(tmp_path):
    model = _FakeViT()
    split = _FakeSplit()
    p = task_prototypes(model, split, 0, C, torch.device("cpu"), batch_size=8)
    assert p.shape == (C, D), f"原型形状应为 ({C},{D})，实得 {tuple(p.shape)}"
    # `ncm_cil_eval` 的口径是「均值再 L2 归一化」，这里必须一致，否则余弦邻居会偏
    assert torch.allclose(p.norm(dim=1), torch.ones(C), atol=1e-5)


def test_prototypes_use_only_the_current_task(tmp_path):
    """任务 1 的原型必须与任务 0 的数据无关：只改任务 0 的样本，任务 1 原型不变。"""
    split = _FakeSplit()
    model = _FakeViT()
    before = task_prototypes(model, split, 1, C, torch.device("cpu"), batch_size=8)
    split._data[0] = (split._data[0][0] + 100.0, split._data[0][1])
    after = task_prototypes(model, split, 1, C, torch.device("cpu"), batch_size=8)
    assert torch.allclose(before, after), "任务 1 的原型被任务 0 的数据影响了"


def test_final_eval_detects_a_shifted_terminal_model(tmp_path):
    """终态模型把旧任务特征整体挪开时，用**冻结原型**评估必须掉分。

    若 `strict_final_eval` 偷偷用终态模型重算原型，分数会保持在高位——这条会红。
    这正是 E5 要量化的那个「评估期访问历史数据」的代价。
    """
    split = _FakeSplit()
    st_model = _FakeViT(shift=0.0)          # 存原型时的模型
    for t in range(T):
        proto = task_prototypes(st_model, split, t, C, torch.device("cpu"), batch_size=8)
        torch.save(proto, _proto_dir(tmp_path) / f"ncm_proto_task{t}.pt")

    same = strict_final_eval(_FakeViT(shift=0.0), split, _proto_dir(tmp_path),
                             C, T, torch.device("cpu"))
    moved = strict_final_eval(_FakeViT(shift=6.0), split, _proto_dir(tmp_path),
                              C, T, torch.device("cpu"))
    assert same[0] > 0.9, f"同一模型下 CIL 应接近 1，实得 {same[0]:.3f}"
    assert moved[0] < same[0] - 0.2, (
        f"终态模型偏移后 CIL 仍为 {moved[0]:.3f}（未偏移 {same[0]:.3f}）——"
        f"说明原型被重算了，E5 测的就不是「冻结原型」的代价")


def test_missing_proto_raises_instead_of_recomputing(tmp_path):
    split = _FakeSplit()
    with pytest.raises(FileNotFoundError):
        strict_final_eval(_FakeViT(), split, tmp_path, C, T, torch.device("cpu"))


def test_per_task_til_matches_cil_when_single_task(tmp_path):
    """只有一个任务时，CIL 与 TIL 的候选集相同，两个数必须相等。"""
    split = _FakeSplit()
    model = _FakeViT()
    for t in range(1):
        torch.save(task_prototypes(model, split, t, C, torch.device("cpu"), batch_size=8),
                   tmp_path / f"ncm_proto_task{t}.pt")
    cil, til, _ = strict_final_eval(model, split, tmp_path, C, 1, torch.device("cpu"))
    assert abs(cil - til) < 1e-6
