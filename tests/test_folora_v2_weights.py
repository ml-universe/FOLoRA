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
