"""InfLoRA 子空间构造的单元测试（不依赖真实数据/GPU）。

核心不变量：
1. 设计出的降维矩阵 B_t 行空间标准正交；
2. B_t ⊆ N_t（新任务输入空间）—— 投影后仍落在当前任务输入张成的空间内；
3. B_t ⊥ M_t（旧任务输入空间）—— 对旧任务无干扰；
4. 多任务累积后 old_basis 保持标准正交且维度受控。
"""

import torch

from peft_cl.methods.inflora import _input_basis, _orth_basis


def _orth_err(M):
    return (M.t() @ M - torch.eye(M.shape[1])).abs().max().item()


def test_input_basis_orthonormal_and_in_row_space():
    torch.manual_seed(0)
    d, r, n = 64, 4, 500
    H = torch.randn(n, d)
    basis = _input_basis(H, r)
    assert basis.shape == (d, r)
    assert _orth_err(basis) < 1e-5

    # basis 必须落在 H 的行空间内
    H_basis = _orth_basis(torch.linalg.svd(H, full_matrices=False)[2].t())
    resid = basis - H_basis @ (H_basis.t() @ basis)
    assert resid.abs().max().item() < 1e-4


def test_basis_is_orthogonal_to_old_subspace_but_keeps_new_directions():
    """旧任务占据前 8 个坐标轴，新任务输入 = 旧方向 + 新方向 8..14。

    投影后应：⊥ 旧空间，且仍有 r 维（新方向有余量）。
    """
    torch.manual_seed(0)
    d, r, n = 64, 4, 500
    old_space = torch.eye(d)[:, :8]
    H = torch.randn(n, 8) @ old_space.t() + torch.randn(n, 6) @ torch.eye(d)[:, 8:14].t()

    basis = _input_basis(H, r)
    proj = _orth_basis(basis - old_space @ (old_space.t() @ basis))
    assert proj.shape[1] == r
    assert (old_space.t() @ proj).abs().max().item() < 1e-4
    assert _orth_err(proj) < 1e-5


def test_accumulated_basis_stays_orthonormal():
    """模拟逐任务累积：每任务并入 r 个新方向，old_basis 应始终标准正交。"""
    torch.manual_seed(0)
    d, r = 128, 8
    M = None
    for _ in range(6):
        H = torch.randn(300, d)
        basis = _input_basis(H, r)
        if M is not None:
            basis = _orth_basis(basis - M @ (M.t() @ basis))
        M = basis if M is None else _orth_basis(torch.cat([M, basis], dim=1))
        assert _orth_err(M) < 1e-4
        # 维度受控：d=128 空间里最多 128 维，不应发散
        assert M.shape[1] <= d
