"""MultiLoRALinear 聚合方式单元测试（纯 CPU）。

为什么值得单独测：`olora_aggregate` 的 sum/mean 之差决定了 O-LoRA 那一行怎么写进
论文主表（sum 是论文原式，20 任务下 ΔW 秩上限 320/768≈42%，NCM 只有 64.79；
mean 是公平性对照）。这个开关如果静默失效（比如 scale 没除、或除错地方），
我们会得到一条错误的对照曲线，而且**不会报错**。
"""

import pytest
import torch
import torch.nn as nn

from peft_cl.adapters.multi_lora import MultiLoRALinear, inject_multi_lora


def make(rank=2, alpha=4, aggregate="sum", n_tasks=3, base=None):
    """建一个带 n_tasks 个非零 adapter 的 MultiLoRALinear。

    base 可传入以让两个 aggregate 变体共用同一个冻结主干——否则两者基座权重不同，
    「mean == sum/3」这种对照根本没法比（第一版就踩了这个）。
    """
    if base is None:
        base = nn.Linear(4, 6)
    m = MultiLoRALinear(base, rank=rank, alpha=alpha, aggregate=aggregate)
    for _ in range(n_tasks):
        a = m.add_task()
        # B 初始化为 0（同 _Adapter），手动给非零值以便观察累加
        with torch.no_grad():
            a.lora_A.weight.normal_()
            a.lora_B.weight.normal_()
    return base, m


def test_unknown_aggregate_raises():
    """拼错的 aggregate 必须当场报错，不能静默 fallback 到 sum。"""
    with pytest.raises(ValueError, match="olora_aggregate"):
        MultiLoRALinear(nn.Linear(4, 6), rank=2, alpha=4, aggregate="average")


def test_no_adapters_returns_base():
    """还没加任务时 forward 应等于冻结主干（早期版本会走空循环，结果相同但没有提前返回）。"""
    base = nn.Linear(4, 6)
    m = MultiLoRALinear(base, rank=2, alpha=4)
    x = torch.randn(3, 4)
    assert torch.allclose(m(x), base(x))


def test_mean_is_sum_divided_by_n_tasks():
    """mean 的定义就是 sum 的输出把增量部分除以任务数——增量不为零时两者必须不同。

    这是对照实验的关键性质：若实现漏了除法，两边数值相等，消融就变成空转。
    """
    torch.manual_seed(0)
    # **共用同一个 base**：两个变体必须只在聚合方式上不同，否则比的是别的差异
    base = nn.Linear(4, 6)
    _, m_sum = make(aggregate="sum", n_tasks=3, base=base)
    _, m_mean = make(aggregate="mean", n_tasks=3, base=base)

    # 让两个模块的 adapter 权重一致（make() 各调用一次 normal_，需手动同步）
    for a_s, a_m in zip(m_sum.adapters, m_mean.adapters):
        a_m.lora_A.weight.data.copy_(a_s.lora_A.weight.data)
        a_m.lora_B.weight.data.copy_(a_s.lora_B.weight.data)

    x = torch.randn(3, 4)
    base_out = base(x)
    delta_sum = m_sum(x) - base_out
    delta_mean = m_mean(x) - base_out

    assert not torch.allclose(delta_sum, delta_mean), "sum 与 mean 不该相等"
    assert torch.allclose(delta_mean, delta_sum / 3, atol=1e-5)
    # 论文原式 sum 的增量为 mean 的 3 倍
    assert torch.allclose(delta_sum, 3 * delta_mean, atol=1e-5)


def test_scale_is_alpha_over_rank_for_sum_but_unchanged_meaning_for_mean():
    """sum 下增量 = (alpha/rank) Σ B_iA_i；mean 下再除任务数。

    用一个手算可验证的例子钉死：rank=1、A 与 B 全 1、单任务、x 全 1，
    则 B Ax 的每个元素 = rank * in_dim = 1*4 = 4，再乘 scale=alpha/rank=4 -> 16。
    """
    base = nn.Linear(4, 6, bias=True)
    with torch.no_grad():
        base.weight.zero_()
        base.bias.zero_()
    m = MultiLoRALinear(base, rank=1, alpha=4)   # scale = 4
    a = m.add_task()
    with torch.no_grad():
        a.lora_A.weight.fill_(1.0)   # (1, 4)
        a.lora_B.weight.fill_(1.0)   # (6, 1)
    x = torch.ones(1, 4)
    # A x = 4 -> B(4) = 4 -> scale * 4 = 16
    assert torch.allclose(m(x), torch.full((1, 6), 16.0), atol=1e-5)


def test_inject_multi_lora_threads_aggregate():
    """inject_multi_lora 必须把 aggregate 传下去（漏传会让所有 run 都用 sum）。"""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 6))

        def forward(self, x):
            return self.mlp(x)

    model = Tiny()
    injected = inject_multi_lora(model, rank=2, alpha=4, targets=("mlp.1",),
                                 aggregate="mean")
    assert len(injected) == 1
    assert injected[0].aggregate == "mean"
    assert model.mlp[1] is injected[0]


def test_add_task_sets_requires_grad_only_on_new_adapter():
    """基座永远冻结；新 adapter 默认可训练，历史 adapter 由方法层再冻结。"""
    base = nn.Linear(4, 6)
    m = MultiLoRALinear(base, rank=2, alpha=4)
    assert not any(p.requires_grad for p in base.parameters())
    a1 = m.add_task()
    a2 = m.add_task()
    assert all(p.requires_grad for p in a2.parameters())
    assert a1 is not a2
