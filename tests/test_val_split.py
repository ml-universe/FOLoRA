"""护栏：`val_frac` 的留出切分必须「逐类、确定、互斥」，且默认值 0 时不动任何东西。

为什么值得一条独立测试
----------------------
`scripts/val_select.py` 的唯一结论是「用验证集重选出来的 λ」。这条结论成立的前提是
三件事同时为真，而它们**错了都不会报错**：

  1. 留出样本与训练样本**互斥**——否则验证集分数等于自评，重选就变成一句空话；
  2. 同一 seed 下**逐类**留出同样的比例——否则候选配置之间比较的不是同一个验证集，
     选出来的 λ 差异里混进了「验证集本身不同」这一项；
  3. `val_frac=0`（默认）与**改动前**逐位一致——否则本仓库已有的全部结果都不再可复现。

判据因此不是「脚本能跑」，而是上面三条逐条成立。
"""

import random

from peft_cl.data.split import ContinualSplit


class _FakeBase:
    """最小数据集：每类 n_per 个样本，标签存在 .targets 里，__getitem__ 回 (索引, 标签)。"""

    def __init__(self, n_classes: int, n_per: int):
        self.targets = [c for c in range(n_classes) for _ in range(n_per)]
        self.transform = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return i, self.targets[i]


def _split(val_frac=0.0, seed=0, n_classes=10, n_per=50, num_tasks=2):
    base_tr = _FakeBase(n_classes, n_per)
    base_te = _FakeBase(n_classes, n_per)
    order = [list(range(i * n_classes // num_tasks, (i + 1) * n_classes // num_tasks))
             for i in range(num_tasks)]
    return ContinualSplit(base_tr, base_te, order, val_frac=val_frac, seed=seed)


def test_default_val_frac_changes_nothing():
    """val_frac=0 时，训练集就是该任务的全部样本（与改动前一致）。"""
    s = _split(val_frac=0.0)
    ds = s.task_dataset(0, "train")
    assert len(ds) == 5 * 50
    assert len(s.val_dataset(0)) == 0


def test_val_and_train_are_disjoint_and_complete():
    """留出的与留下的互斥，且合起来正好是原训练集（无丢失、无重复）。"""
    s = _split(val_frac=0.1)
    tr = set(s.task_dataset(0, "train").indices)
    va = set(s.val_dataset(0).indices)
    assert tr.isdisjoint(va)
    assert tr | va == set(s._task_indices(0, "train"))


def test_val_fraction_is_per_class():
    """逐类留出：每个类在验证集里都恰好占 round(n_per * val_frac) 个。"""
    s = _split(val_frac=0.1)
    ds = s.val_dataset(0)
    labels = [s.base_train.targets[i] for i in ds.indices]
    counts = {c: labels.count(c) for c in set(labels)}
    assert set(counts.values()) == {5}, counts          # 50 * 0.1 = 5
    assert len(counts) == 5                             # 任务 0 的五个类一个不落


def test_split_is_deterministic_and_seeded():
    """同 seed 同结果；换 seed 换一批（否则重选实验里不同 seed 共享同一验证集）。"""
    a = set(_split(val_frac=0.1, seed=0).val_dataset(0).indices)
    b = set(_split(val_frac=0.1, seed=0).val_dataset(0).indices)
    c = set(_split(val_frac=0.1, seed=1).val_dataset(0).indices)
    assert a == b
    assert a != c


def test_tasks_have_independent_holds():
    """不同任务留出的是各自的样本（不会因为 task 编号相同就复用同一批索引）。"""
    s = _split(val_frac=0.1)
    v0 = set(s.val_dataset(0).indices)
    v1 = set(s.val_dataset(1).indices)
    assert v0.isdisjoint(v1)
