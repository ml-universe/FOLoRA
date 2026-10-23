"""界探针（peft_cl/utils/bound_probe.py）单元测试，纯 CPU。

为什么值得单独测
----------------
探针挂在**训练循环内部**，而且会临时改写 adapter 权重再还原。两个失效模式都很隐蔽：

1. **还原失败**：探针扫完 α 后若没把 adapter 写回 θ_t，训练会带着一个错的权重继续跑完
   剩余任务——结果文件看起来完全正常，只是数字全错。没有任何报错。
2. **拼序错位**：`flatten_adapter` 的拼接顺序必须与 FOLoRAv2Method._flatten 一致
   （A 在前、B 在后），否则 δθ 是乱的，但二阶项仍会算出一个「看起来合理」的数。

所以这里测的是**不变量**（还原、往返、量纲），而不是某个具体数值。
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from peft_cl.adapters.lora import inject_lora
from peft_cl.utils.bound_probe import (ALPHAS, flatten_adapter, probe_bound_terms,
                                       unflatten_into)


class Tiny(nn.Module):
    """带 mlp.3 的最小模型，好让 inject_lora 的默认 target 命中。"""

    def __init__(self, in_dim=6, n_cls=5):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 6), nn.Linear(6, 6),
                                 nn.Linear(6, 6), nn.Linear(6, n_cls))

    def forward(self, x):
        return self.mlp(x)


class FakeSplit:
    """给出可 DataLoader 的小数据集；标签是全局类索引（探针按全局标签算 CE）。"""

    def __init__(self, n_cls=5, n=8):
        torch.manual_seed(0)
        self.x = torch.randn(n, 6)
        self.y = torch.randint(0, n_cls, (n,))

    def task_dataset(self, task_id, split):
        return TensorDataset(self.x, self.y)


class FakeMethod:
    def __init__(self, loras, rank=2):
        self.loras = loras
        self.acc_grads = {}
        self.ref_params = {}
        for i, lora in enumerate(loras):
            d = flatten_adapter(lora).numel()
            # 行范数故意不全相等：若探针只用了行数或忽略了权重，二阶项会算错
            G = torch.randn(3, d) * torch.linspace(1.0, 2.0, 3).unsqueeze(1)
            self.acc_grads[i] = G
            self.ref_params[i] = flatten_adapter(lora).detach().clone()


def build(seed=0):
    torch.manual_seed(seed)
    model = Tiny()
    loras = inject_lora(model, rank=2, alpha=2, targets=("mlp.3",))
    method = FakeMethod(loras)
    # 让「当前」adapter 与 ref_params 不同，否则 δθ = 0、二阶项恒为 0，测不出东西
    with torch.no_grad():
        for lora in loras:
            lora.lora_A.weight.normal_()
            lora.lora_B.weight.normal_()
    return model, method, loras


def test_flatten_unflatten_roundtrip():
    """往返必须逐位相等，且顺序是 A 在前 B 在后（与 _flatten 的约定一致）。"""
    _, _, loras = build()
    lora = loras[0]
    flat = flatten_adapter(lora)
    assert flat.numel() == lora.lora_A.weight.numel() + lora.lora_B.weight.numel()
    # A 段
    n_a = lora.lora_A.weight.numel()
    assert torch.allclose(flat[:n_a], lora.lora_A.weight.reshape(-1))
    assert torch.allclose(flat[n_a:], lora.lora_B.weight.reshape(-1))

    with torch.no_grad():
        lora.lora_A.weight.zero_()
        lora.lora_B.weight.zero_()
    unflatten_into(lora, flat)
    assert torch.allclose(flatten_adapter(lora), flat)


def test_probe_restores_adapter_exactly():
    """**最关键的性质**：探针扫完 α 后 adapter 必须精确还原成 θ_t。

    不还原不会报错，只会让后续训练从一个错误权重继续——结果文件依然「正常」。
    """
    model, method, loras = build()
    before = [flatten_adapter(l).detach().clone() for l in loras]
    rec = probe_bound_terms(model, method, FakeSplit(), torch.device("cpu"), task_id=1)
    assert rec is not None
    after = [flatten_adapter(l).detach().clone() for l in loras]
    for b, a in zip(before, after):
        assert torch.equal(b, a), "探针没有把 adapter 还原成 θ_t"


def test_probe_restores_train_mode():
    """探针内部切到 eval 测损失，结束后必须恢复调用前的训练/评估模式。"""
    model, method, _ = build()
    model.train()
    probe_bound_terms(model, method, FakeSplit(), torch.device("cpu"), task_id=1)
    assert model.training is True
    model.eval()
    probe_bound_terms(model, method, FakeSplit(), torch.device("cpu"), task_id=1)
    assert model.training is False


def test_quadratic_term_matches_bruteforce():
    """二阶项必须是 ½·Σ_l ‖G_l δθ_l‖²，且随 δθ 严格为零时归零。"""
    model, method, loras = build()
    rec = probe_bound_terms(model, method, FakeSplit(), torch.device("cpu"), task_id=1)

    cur = [flatten_adapter(l).detach().float() for l in loras]
    delta = [c - method.ref_params[i].float() for i, c in enumerate(cur)]
    manual = 0.5 * sum(float((method.acc_grads[i].float() @ delta[i]).pow(2).sum())
                       for i in range(len(loras)))
    assert rec["quad_term"] == pytest.approx(manual, rel=1e-6)
    assert rec["quad_term"] > 0
    # δθ 的范数应与手工计算一致
    assert rec["delta_norm"] == pytest.approx(
        float(torch.cat(delta).norm()), rel=1e-5)


def test_probe_returns_none_when_not_applicable():
    """第一个任务没有「之前的任务」，界的 LHS 为空；缺 acc_grads/ref_params 也不能硬算。"""
    model, method, _ = build()
    assert probe_bound_terms(model, method, FakeSplit(), torch.device("cpu"),
                             task_id=0) is None

    empty = FakeMethod(method.loras)
    empty.acc_grads, empty.ref_params = {}, {}
    assert probe_bound_terms(model, empty, FakeSplit(), torch.device("cpu"),
                             task_id=1) is None


def test_probe_reports_finite_fit_and_curve():
    """α 曲线长度与 α 一致，拟合系数为有限值，dL(0)=0（构造保证）。"""
    model, method, _ = build()
    rec = probe_bound_terms(model, method, FakeSplit(), torch.device("cpu"), task_id=1)
    assert rec["alphas"] == [float(a) for a in ALPHAS]
    assert len(rec["dL_curve"]) == len(ALPHAS)
    assert rec["dL_curve"][0] == 0.0
    for k in ("fit_a_linear", "fit_b_quadratic", "fit_c_cubic"):
        assert np.isfinite(rec[k]), f"{k} 不是有限值"
