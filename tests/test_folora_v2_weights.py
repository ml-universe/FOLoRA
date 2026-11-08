# -*- coding: utf-8 -*-
"""FOLoRA v2 的加权/等权尺度恒等式（论文主方法的第一个直接测试，P1-8②）。

为什么值得单独测：论文 §5.3 的「Fisher 加权 vs 等权」消融之所以能作为**单变量**对照，
完全依赖一个恒等式 —— 等权臂与加权臂的**总惩罚权重相等**，只差「怎么分配」。若不成立，
两臂的差异就分不清是「分配方式」还是「整体尺度」。2026-09-25 之前用算术平均 σ̄ 构造
等权臂，实测总权重比仅 0.248（见 folora_v2.py::_equalize_row_weights 的 docstring），
正是这种混杂。这个文件把那次的教训固化成断言。

数学：G 的每行是 σ_j v_jᵀ，于是
    ‖G δθ‖² = Σ_j σ_j² (v_jᵀ δθ)²
正则惩罚天然按 Fisher 特征值 σ_j² 加权。把每行缩放到公共模长 m 后总权重变成 k·m²，
令 k·m² = Σ_j σ_j² 只需 m = sqrt(mean(σ_j²)) = RMS(σ_j)。恒等式与 σ 的分布无关。

全部为纯张量运算，CPU 可跑，不需要模型、不需要 GPU。
"""

import torch

from peft_cl.methods.folora_v2 import FOLoRAv2Method

equalize = FOLoRAv2Method._equalize_row_weights


def _sigma(G):
    return G.norm(dim=1)


# --------------------------------------------------------------- 恒等式本体

def test_equalize_preserves_total_weight():
    """核心：等权化后总权重 Σ‖row‖² 与加权版 Σσ_j² 相等。"""
    torch.manual_seed(0)
    for k, d in [(16, 128), (64, 512), (4, 32), (128, 64)]:
        G = torch.randn(k, d)
        G_eq = equalize(G.clone())
        total_weighted = (G.norm(dim=1) ** 2).sum()
        total_equal = (G_eq.norm(dim=1) ** 2).sum()
        assert torch.isclose(total_equal, total_weighted, rtol=1e-5), (
            f"k={k}, d={d}: 等权总权重 {total_equal:.6f} != 加权总权重 {total_weighted:.6f}"
        )


def test_identity_holds_when_sigma_are_wildly_unequal():
    """真正的考验：σ 分布越悬殊，算术平均版错得越离谱，RMS 版必须仍然精确。

    用实测量级（σ ∈ [1.94, 19.03]）造一个长尾谱。
    """
    torch.manual_seed(1)
    k, d = 64, 256
    # 让行范数跨度接近实测的 ~10x
    scale = torch.linspace(1.94, 19.03, k).unsqueeze(1)
    G = scale * torch.randn(k, d)
    sig = _sigma(G)
    assert sig.max() / sig.min() > 5.0, "前提没构造出来：σ 不够悬殊"

    G_eq = equalize(G.clone())
    assert torch.isclose((G_eq.norm(dim=1) ** 2).sum(),
                         (sig ** 2).sum(), rtol=1e-5)


def test_all_rows_share_one_common_norm():
    """等权化必须把每行缩放到**同一个**模长，否则「等权」名不副实。"""
    torch.manual_seed(2)
    G = torch.randn(32, 128)
    G_eq = equalize(G.clone())
    norms = G_eq.norm(dim=1)
    assert torch.allclose(norms, norms[0].expand_as(norms), rtol=1e-4)


def test_directions_are_unchanged():
    """只改尺度、不改方向：每行仍是原行的正数倍，故 v_jᵀδθ 不受影响。"""
    torch.manual_seed(3)
    G = torch.randn(16, 64)
    G_eq = equalize(G.clone())
    ratio = (G_eq.norm(dim=1) / G.norm(dim=1))
    # 逐行验证 G_eq[i] = ratio[i] * G[i]，ratio > 0
    assert torch.all(ratio > 0)
    assert torch.allclose(G_eq, G * ratio.unsqueeze(1), rtol=1e-5)
    # 方向余弦为 1
    cos = (G_eq * G).sum(dim=1) / (G_eq.norm(dim=1) * G.norm(dim=1))
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


# ------------------------------------------------- 回归护栏：旧构造必须失败

def test_arithmetic_mean_construction_would_not_satisfy_the_identity():
    """把旧的「算术平均」构造钉在这里：它必然**不**满足恒等式。

    这不是在测产物，而是在测「这个测试本身有鉴别力」——如果哪天有人把实现改回
    算术平均，上面那条 test_equalize_preserves_total_weight 会红，这条解释原因。
    """
    torch.manual_seed(4)
    k, d = 64, 256
    scale = torch.linspace(1.94, 19.03, k).unsqueeze(1)
    G = scale * torch.randn(k, d)

    sig = _sigma(G)
    old = G / sig.unsqueeze(1) * sig.mean()          # 算术平均版（已废弃）
    ratio = (old.norm(dim=1) ** 2).sum() / (sig ** 2).sum()

    assert ratio < 0.9, (
        f"算术平均版的总权重比 {ratio:.3f} 竟然接近 1 —— 说明这个构造能成立，"
        "那么 2026-09-25 的修正就没必要，本文件的前提需要重新审视"
    )


# ------------------------------------------------- 惩罚项 = 特征值加权重叠

def test_penalty_equals_eigenvalue_weighted_overlap():
    """‖G δθ‖² 必须等于 Σ_j σ_j² (v_jᵀ δθ)² —— 即惩罚真的按 σ_j² 加权。"""
    torch.manual_seed(5)
    k, d = 16, 64
    G = torch.randn(k, d)
    dtheta = torch.randn(d)

    lhs = (G @ dtheta).pow(2).sum()

    sig = _sigma(G)
    # v_j = row_j / σ_j，故 v_jᵀ δθ = (row_j · δθ) / σ_j
    v_dot = (G @ dtheta) / sig
    rhs = (sig.pow(2) * v_dot.pow(2)).sum()

    assert torch.isclose(lhs, rhs, rtol=1e-5), f"{lhs:.6f} != {rhs:.6f}"


def test_equalized_penalty_is_the_flat_version():
    """等权臂的惩罚 = 用公共模长 m 替掉全部 σ_j，即「同总权重、不同分配」。"""
    torch.manual_seed(6)
    k, d = 32, 128
    G = torch.randn(k, d)
    dtheta = torch.randn(d)

    G_eq = equalize(G.clone())
    lhs = (G_eq @ dtheta).pow(2).sum()

    sig = _sigma(G)
    m = sig.pow(2).mean().sqrt()
    v_dot = (G @ dtheta) / sig
    rhs = (m ** 2 * v_dot.pow(2)).sum()

    assert torch.isclose(lhs, rhs, rtol=1e-5), f"{lhs:.6f} != {rhs:.6f}"


# =====================================================================================
# 2026-11-04：三模式同尺度（R7 rec4 的 random-subspace 与 ordinary-displacement 对照）
# =====================================================================================
# 这两条对照只有在**同一个 λ**下才有鉴定力。若尺度没对齐，比较出来的就是「正则强度」
# 而不是「方向来源 / 有没有投影」。2026-11-04 的初版就写错过一次：isotropic 系数写成
# mean(σ²) 而非 Σσ²/d，与另两臂差 k/d ≈ 1000 倍，等于 none 臂根本没加正则——
# 而那会被误读成「投影没有贡献」。下面把正确的期望值钉死。

import types

import pytest


def _bare_method(seed: int = 0):
    """构造一个不依赖模型与 GPU 的方法实例（只测标量/张量算式）。"""
    m = object.__new__(FOLoRAv2Method)
    m.config = types.SimpleNamespace(seed=seed, folora_proj="fisher",
                                     folora_weighted=True, folora_lambda=3.0)
    m._rand_basis, m._rand_dev, m._iso_dev, m._acc_dev = {}, {}, {}, {}
    return m


def _mean_isotropic_penalty(M, delta_count=400, seed=11):
    """对大量各向同性单位位移取平均，得到该模式的期望惩罚。"""
    d = M.shape[1]
    g = torch.Generator().manual_seed(seed)
    acc = 0.0
    for _ in range(delta_count):
        v = torch.randn(d, generator=g)
        v = v / v.norm()
        acc += float((M @ v).pow(2).sum())
    return acc / delta_count


def test_three_modes_have_matching_expected_penalty():
    """fisher / random / none 三臂对**各向同性位移**的期望惩罚必须相等。

    三者理论值同为 (Σσ_j²)/d：Fisher 因为 E[(v_jᵀδθ)²] = ‖δθ‖²/d；
    random 因为 k 个正交方向同理；none 是显式取的这个系数。
    """
    torch.manual_seed(10)
    k, d = 64, 4096
    G = torch.randn(k, d)
    m = _bare_method()

    theory = float((G.norm(dim=1) ** 2).sum() / d)

    R = m._random_directions(0, G, torch.device("cpu"))
    iso = float(m._isotropic_scale(0, G))

    got = {
        "fisher": _mean_isotropic_penalty(G),
        "random": _mean_isotropic_penalty(R),
        "none": iso,
    }
    for name, val in got.items():
        assert abs(val / theory - 1.0) < 0.05, (
            f"{name} 臂期望惩罚 {val:.4f} 与理论值 {theory:.4f} 相对偏差 "
            f"{abs(val / theory - 1):.3f} 超过 5% —— 三模式不同尺度，消融会被混杂")


def test_isotropic_scale_is_not_the_mean_sigma_squared():
    """回归护栏：钉住 2026-11-04 写错的那个值（mean(σ²)，差 k/d 倍）。

    如果哪天有人把实现改回 `norm.pow(2).mean()`，这条会红并指出正确的量。
    """
    torch.manual_seed(12)
    k, d = 64, 61440
    G = torch.randn(k, d)
    m = _bare_method()

    correct = float(G.norm(dim=1).pow(2).sum() / d)
    wrong = float(G.norm(dim=1).pow(2).mean())
    ratio = correct / wrong
    assert abs(ratio - k / d) < 1e-6, f"k/d 应为 {k / d:.6f}，实得 {ratio:.6f}"

    got = float(m._isotropic_scale(0, G))
    assert abs(got / correct - 1.0) < 1e-6, (
        f"isotropic 系数 {got:.4f} 应为 Σσ²/d = {correct:.4f}，"
        f"若等于 {wrong:.4f} 说明退回了差 {d / k:.0f} 倍的旧写法")


def test_random_basis_is_fixed_within_a_run():
    """随机子空间在**整个 run 内固定**：同层重复调用必须逐位相同，否则它不是
    「一个随机子空间」而是「每任务换一个」，对照就不成立了。"""
    torch.manual_seed(13)
    G1 = torch.randn(16, 256)
    G2 = torch.randn(16, 256)     # 模拟下一个任务（σ 尺度不同）
    m = _bare_method()
    R1 = m._random_directions(0, G1, torch.device("cpu"))
    m._rand_dev.clear()           # 模拟任务切换后重新缩放
    R2 = m._random_directions(0, G2, torch.device("cpu"))

    # 方向必须一致（归一化后比较），尺度按各自任务重算
    cos = (R1 / R1.norm(dim=1, keepdim=True) *
           R2 / R2.norm(dim=1, keepdim=True)).sum(dim=1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-4), \
        "随机基在不同任务间变了 —— 对照退化"


def test_random_basis_depends_on_seed_and_layer():
    """不同 seed / 不同层必须得到不同方向，否则 10 个种子的「独立重抽」是假的。"""
    G = torch.randn(8, 64)
    R_a = _bare_method(seed=0)._random_directions(0, G, torch.device("cpu"))
    R_b = _bare_method(seed=1)._random_directions(0, G, torch.device("cpu"))
    R_c = _bare_method(seed=0)._random_directions(1, G, torch.device("cpu"))
    assert not torch.allclose(R_a, R_b), "换 seed 方向没变"
    assert not torch.allclose(R_a, R_c), "换层序号方向没变"


# ------------------------------------------- 三个模式在真实方法对象里跑通（CPU 极简模型）

import torch.nn as nn

from peft_cl.utils.config import CLConfig


def _tiny_folora(proj: str, seed: int = 0):
    """建一个真的 FOLoRAv2Method（含 LoRA 注入），手工塞进 ref_params 与 acc_grads。

    不走 after_task：那需要真实数据与前向。这里只验证 `regularization_loss()` 的
    三个分支都能**在真实方法对象上**跑通、返回有限正数、且梯度能回到 LoRA 参数——
    这正是 run_lambda_grid 里 `--folora_proj random/none` 会走的那段代码。
    """
    from peft_cl.methods.folora_v2 import FOLoRAv2Method as M
    torch.manual_seed(seed)
    model = nn.Module()
    model.mlp = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                              nn.Linear(8, 4))          # inject_lora 默认注入点 mlp.3
    cfg = CLConfig(method="folora_v2", lora_rank=2, lora_alpha=2,
                   folora_lambda=3.0, folora_topk=4, folora_proj=proj, seed=seed)
    method = M(model, cfg)

    for i, lora in enumerate(method.loras):
        theta = method._flatten(lora).detach()
        method.ref_params[i] = theta.clone()
        # 轻微移动，让 δθ ≠ 0
        with torch.no_grad():
            lora.lora_A.weight.add_(torch.randn_like(lora.lora_A.weight) * 0.05)
            lora.lora_B.weight.add_(torch.randn_like(lora.lora_B.weight) * 0.05)
        method.acc_grads[i] = torch.randn(4, theta.numel())
    return model, method, cfg


@pytest.mark.parametrize("proj", ["fisher", "random", "none"])
def test_regularization_loss_runs_for_every_proj_mode(proj):
    """三个模式都必须给出有限正惩罚，且梯度能回到可训练参数上。"""
    model, method, cfg = _tiny_folora(proj)
    loss = method.regularization_loss()
    assert torch.isfinite(loss).all() and float(loss.detach()) > 0.0, \
        f"proj={proj}: 惩罚 {loss} 非有限正数"
    loss.backward()
    grads = [p.grad for p in method.loras[0].parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), \
        f"proj={proj}: 梯度没回到 LoRA 参数"


def test_none_mode_uses_the_isotropic_coefficient_on_the_real_method():
    """none 模式的系数必须等于 Σσ²/d（在真实方法对象上再核对一次）。"""
    model, method, cfg = _tiny_folora("none")
    i = 0
    G = method.acc_grads[i]
    expected = float(G.norm(dim=1).pow(2).sum() / G.shape[1])
    got = float(method._isotropic_scale(i, G))
    assert abs(got / expected - 1.0) < 1e-6
