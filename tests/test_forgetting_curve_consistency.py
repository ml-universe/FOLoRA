"""遗忘曲线（图）与主表指标（`metrics.forgetting`）的口径一致性测试。

背景：`scripts/plot_forgetting_curve.py:forgetting_curve()` 与
`src/peft_cl/metrics/metrics.py:forgetting()` 是同一物理量的两份实现，虽然脚本
docstring 承诺「末点就是主表 FGT 列」，但两者曾因分母不同（t 项 vs T 项，差
T/(T-1)，T=20 时 5.26%）而对不上，且该函数没有任何测试覆盖，bug 长期未被发现
（2026-10-03 修复）。本测试把这个不变量钉住。

容差说明：`metrics.forgetting` 走 torch float32，曲线走 numpy float64，末点比较
的残差在 1e-7 量级，取 1e-5 作为安全裕度——**远小于**任何真实的口径偏差（5% 量级），
所以本测试能真正拦住回归。
"""

import numpy as np
import pytest

from peft_cl.metrics.metrics import forgetting
from scripts.plot_forgetting_curve import accuracy_curve, forgetting_curve

TOL = 1e-5


def _random_matrix(seed: int, T: int = 20) -> np.ndarray:
    """造一个上三角有值的 20x20 准确率矩阵（acc[t][j]，j>t 为 0，未被使用）。"""
    rng = np.random.default_rng(seed)
    acc = np.zeros((T, T))
    for t in range(T):
        acc[t, : t + 1] = rng.uniform(0.3, 0.95, size=t + 1)
    return acc


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_curve_last_point_equals_table_metric(seed):
    """曲线末点 t=T-1 必须逐位等于主表的 FGT 标量。"""
    acc = _random_matrix(seed)
    curve_last = float(forgetting_curve(acc)[-1])
    table = float(forgetting(acc.tolist()))
    assert abs(curve_last - table) < TOL, (
        f"曲线末点 {curve_last:.10f} != 主表 FGT {table:.10f}"
        f"（差 {abs(curve_last - table):.3e}）；两者口径又漂了")


def test_curve_matches_metric_on_prefix():
    """把前 t+1 个任务单独喂给 metrics.forgetting，应等于曲线在时刻 t 的值。

    这验证的不只是末点，而是「曲线是 metrics.forgetting 的逐步版本」这一更强的不变量。
    """
    acc = _random_matrix(7)
    curve = forgetting_curve(acc)
    for t in range(1, acc.shape[0]):
        sub = acc[: t + 1, : t + 1].tolist()
        assert abs(curve[t] - float(forgetting(sub))) < TOL, f"时刻 t={t} 口径不一致"


def test_accuracy_curve_last_point_is_final_average():
    """准确率曲线末点应等于主表 ACC 列（最后一行的均值）。"""
    acc = _random_matrix(11)
    assert abs(float(accuracy_curve(acc)[-1]) - float(acc[-1].mean())) < TOL


def test_curve_ignores_upper_triangle_garbage():
    """j>t 的格子（未填充）不得影响结果；脚本不应读到它们。"""
    acc = _random_matrix(13)
    polluted = acc.copy()
    polluted[np.triu_indices(acc.shape[0], k=1)] = 999.0   # 上三角塞垃圾
    assert np.allclose(forgetting_curve(acc), forgetting_curve(polluted), equal_nan=True)
    assert np.allclose(accuracy_curve(acc), accuracy_curve(polluted))
