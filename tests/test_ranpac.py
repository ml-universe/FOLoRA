# -*- coding: utf-8 -*-
r"""RanPAC 复现（`scripts/eval_ranpac.py`）的语义测试。

为什么必须在出数前测
--------------------
这里面的错误**一个都不会报错**，只会给出一个「看着合理」的准确率：

  * 类号重映射差一位 → 每个任务的测试集都错位，ACC 仍是个像样的数；
  * `predict_seen` 忘了截断候选集 → 没学过的类也在候选里，分类器等于偷看答案；
  * one-hot 的列数用了「已见类数」而不是「全局类数」→ 前几个任务看着更高，
    因为 $Q$ 的维度随任务增长，先学的类权重被后来的列挤掉；
  * 岭系数的 80/20 切分没打乱 → 变成「按类切分」，选出的 $\lambda$ 没有意义。

所以这里构造一个「类间距离 >> 类内方差」的小问题，把每条语义钉死：
正确实现的边界是**接近 1.0**，任何一条出错都会明显低于它。
"""

import torch

from peft_cl.data.split import ContinualSplit, make_class_order
from scripts.eval_ranpac import RanPAC, ranpac_matrix, remap_to_global

D = 16          # 特征维度
C = 2           # 每任务类数
T = 4           # 任务数
N = 40          # 每类样本数


def _toy():
    """每类一个固定中心、类内加小噪声。标签用**原始类号**（打乱过的非连续顺序）。"""
    g = torch.Generator().manual_seed(0)
    class_order = make_class_order(T * C, T, seed=3)      # 原始类号被洗过的分任务顺序
    centers = {}
    for gi, c in enumerate([c for task in class_order for c in task]):
        centers[int(c)] = torch.randn(D, generator=g) * 6.0
    tr_f, tr_y, te_f, te_y = [], [], [], []
    for c, ctr in centers.items():
        tr_f.append(ctr + 0.1 * torch.randn(N, D, generator=g))
        tr_y.append(torch.full((N,), c, dtype=torch.long))
        te_f.append(ctr + 0.1 * torch.randn(N // 2, D, generator=g))
        te_y.append(torch.full((N // 2,), c, dtype=torch.long))
    return (class_order, torch.cat(tr_f), torch.cat(tr_y),
            torch.cat(te_f), torch.cat(te_y))


def test_remap_matches_continual_split():
    """重映射必须与 `ContinualSplit.class_to_global` **逐元素**相同。

    这是最容易差一位的地方：`class_to_global` 是「原始类号 → 全局下标」的字典，
    全局下标就是 class_order 展平后的位置。用一份独立的实现交叉核对。
    """
    class_order = make_class_order(100, 20, seed=7)
    split = ContinualSplit(None, None, class_order)
    labels = torch.arange(100)
    got = remap_to_global(labels, class_order)
    want = torch.tensor([split.class_to_global[int(c)] for c in range(100)])
    assert torch.equal(got, want), f"重映射不一致：\n{got.tolist()}\n{want.tolist()}"
    # 每个任务的 k 个**原始**类，其全局下标必须恰好是连续的 [t*k, (t+1)*k)
    # （注意 `got` 是按原始类号索引的，不能直接取 got[t*k:(t+1)*k]）
    k = 100 // 20
    for t in range(20):
        raw = torch.tensor(class_order[t], dtype=torch.long)
        assert sorted(got[raw].tolist()) == list(range(t * k, (t + 1) * k))


def test_unseen_class_weights_are_exactly_zero():
    """只学过前 t 个任务时，未见类的权重行必须**恰好为 0**（$Q$ 的对应列恒为 0）。"""
    class_order, tr_f, tr_y, te_f, te_y = _toy()
    tr_y = remap_to_global(tr_y, class_order)
    head = RanPAC(D, T * C, M=64, rp_seed=0, ridge_select="dual")
    head.update(tr_f[tr_y < C], tr_y[tr_y < C], task_id=0)
    assert torch.all(head.Wo[C:] == 0), "未见类的权重不是 0 —— one-hot 的列数写错了"
    assert torch.any(head.Wo[:C] != 0)


def test_predict_seen_ignores_unseen_columns():
    """`predict_seen(n_seen)` 必须把候选集截断到已见类，否则等于偷看答案。

    这里把未见类的权重行人为撑大：若不截断，它们的 logit 会碾压已见类。
    """
    class_order, tr_f, tr_y, te_f, te_y = _toy()
    tr_y = remap_to_global(tr_y, class_order)
    te_y = remap_to_global(te_y, class_order)
    head = RanPAC(D, T * C, M=64, rp_seed=0, ridge_select="dual")
    head.update(tr_f[tr_y < C], tr_y[tr_y < C], task_id=0)
    with torch.no_grad():
        head.Wo[C:] = 1e3
    pred = head.predict_seen(te_f, n_seen=C)
    assert int(pred.max()) < C, "候选集没截断：预测落到了没学过的类上"
    seen = te_y < C
    acc = float((pred[seen] == te_y[seen]).float().mean())
    assert acc > 0.95, f"只学了两个可分性极强的类，准确率应接近 1，实得 {acc:.3f}"


def test_matrix_is_near_perfect_on_separable_toy():
    """任务数少、类间距离远时，RanPAC 的整张矩阵应接近 1；同时检查下三角填充。"""
    class_order, tr_f, tr_y, te_f, te_y = _toy()
    tr_y = remap_to_global(tr_y, class_order)
    te_y = remap_to_global(te_y, class_order)
    acc, ridge = ranpac_matrix(tr_f, tr_y, te_f, te_y, T, C, M=64, rp_seed=0,
                               ridge_select="dual", verbose=False)
    for t in range(T):
        for j in range(T):
            if j <= t:
                assert acc[t][j] > 0.9, f"acc[{t}][{j}]={acc[t][j]:.3f}，可分问题上应接近 1"
            else:
                assert acc[t][j] == 0.0, f"acc[{t}][{j}] 是未来任务，必须保持 0"
    assert ridge > 0


def test_dual_and_direct_select_the_same_ridge():
    """对偶形式与官方直接形式**数学等价**：小问题上必须选出同一个 λ。

    这条是给「为了跑得快改成对偶」这个决定兜底的：若两者在可控规模上不一致，
    说明其中一份实现写错了，那 10000 维上的结果就不能信。
    """
    class_order, tr_f, tr_y, _te_f, _te_y = _toy()
    tr_y = remap_to_global(tr_y, class_order)
    sel = {}
    for mode in ("dual", "direct"):
        head = RanPAC(D, T * C, M=256, rp_seed=0, ridge_select=mode)
        H = head._phi(tr_f[:80])
        Y = torch.zeros(80, T * C)
        Y[torch.arange(80), tr_y[:80]] = 1.0
        sel[mode] = head.select_ridge(H, Y, shuffle_seed=0)
    assert sel["dual"] == sel["direct"], f"两种实现选出不同 λ：{sel}"
