"""任务边界上的 NCM 准确率行（为「真实遗忘」补证据）。

为什么要它
----------
`scripts/eval_ncm_sweep.py` 只评估**终态** checkpoint：它用 θ_T 提一次特征，然后
在这个特征空间里逐任务累积类均值，得到一张 20×20 矩阵。那张矩阵**不能**用来算遗忘
——`acc[t][j]` 里的 θ 永远是 θ_T，不是任务 t 结束时的 θ_t（本仓库已单独记过这个语义
坑：Fig 2/3 的横轴其实是候选类数，不是时间）。真正的遗忘必须在**每个任务边界**
用**当时的**模型重新提特征、重新建类均值，行与行之间才可比。

    BWT = 1/(T-1) * Σ_{j<T-1} ( a[T-1][j] − a[j][j] )

本模块就是在训练循环里做这一行的：`ncm_row()` 给定「当前模型 + 已经见过的任务」，
返回该边界在 CIL 与 TIL 两种口径下对**全部历史任务**的准确率。

与 `scripts/eval_ncm.py` 的关系
--------------------------------
特征提取与「逐任务累积类均值」的**数值口径必须与主表完全一致**，否则补出来的遗忘
曲线和主表说的不是同一件事。因此这里不另写一套实现，而是直接复用
`scripts.eval_ncm.extract_features`（惰性导入，避免 src 反向依赖 scripts 的加载期问题）。
类均值的构造逐行照抄 `ncm_cil_eval`：L2 归一化特征 → 逐类取均值 → 再归一化 → 余弦最近邻。

成本
----
一个边界要提 (t+1)/T 的全量特征（图像侧与类别侧都算）。20 个边界全开时约为
「一趟全量特征提取」的 10.5 倍，所以配置里给了 stride：只在第 N、2N、… 个边界处算，
最后一个边界永远算（否则拿不到 a[T-1][*]，BWT 无从谈起）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader


def _extract(model, dataset, device, batch_size: int, num_workers: int):
    """[CLS] 特征（heads 之前），fp32、CPU。与 scripts/eval_ncm.extract_features 同口径。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                z = model._process_input(x)
                n = z.shape[0]
                cls = model.class_token.expand(n, -1, -1)
                z = model.encoder(torch.cat([cls, z], dim=1))[:, 0]
            feats.append(z.float().cpu())
            labels.append(y.clone())
    if not feats:
        return torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def ncm_row(model, split, task_id: int, classes_per_task: int, num_tasks: int,
            device, batch_size: int = 64, num_workers: int = 0):
    """返回 (row_cil, row_til)，长度均为 num_tasks，未见的任务位保持 0.0。

    row_cil[j]：任务 j 的测试集上，对**已见全部类**做 NCM 的准确率（类增量口径）。
    row_til[j]：同一批样本，候选类限定为任务 j 自己的 classes_per_task 个类。

    时间语义：调用发生在任务 task_id 训练完之后、`after_task` 之前，所以用的是 θ_t
    （重要性估计尚未把当前任务并进去，但这不影响特征本身——特征只取决于权重）。
    """
    was_training = model.training
    model.eval()
    try:
        tr = ConcatDataset([split.task_dataset(j, "train") for j in range(task_id + 1)])
        te = ConcatDataset([split.task_dataset(j, "test") for j in range(task_id + 1)])
        tr_feats, tr_labels = _extract(model, tr, device, batch_size, num_workers)
        te_feats, te_labels = _extract(model, te, device, batch_size, num_workers)
    finally:
        model.train(was_training)

    trn = F.normalize(tr_feats, dim=1)
    ten = F.normalize(te_feats, dim=1)

    n_seen_classes = (task_id + 1) * classes_per_task
    means = torch.zeros(n_seen_classes, trn.shape[1])
    for c in range(n_seen_classes):
        idx = (tr_labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:                     # 理论上不会发生；宁可显式报错也别静默给 0 均值
            raise RuntimeError(f"NCM trace: class {c} has no training samples seen so far")
        means[c] = F.normalize(trn[idx].mean(dim=0), dim=0)

    row_cil = [0.0] * num_tasks
    row_til = [0.0] * num_tasks
    for j in range(task_id + 1):
        lo, hi = j * classes_per_task, (j + 1) * classes_per_task
        idx = (te_labels >= lo) & (te_labels < hi)
        if not bool(idx.any()):
            continue
        x, y = ten[idx], te_labels[idx]
        # CIL：候选 = 已见全部类（类号 = 下标，不需要平移）
        row_cil[j] = (x @ means.t()).argmax(dim=1).eq(y).float().mean().item()
        # TIL：候选 = 本任务的类（下标要加回 lo 才能对到全局类号）
        sub = means[lo:hi]
        pred = (x @ sub.t()).argmax(dim=1) + lo
        row_til[j] = pred.eq(y).float().mean().item()
    return row_cil, row_til


# =====================================================================================
# 严格序贯原型协议（2026-11-04，回应 R7-W2 / R7 rec2 / R8-10）
# =====================================================================================
# 主表的 NCM 是**终态重放**：用 θ_T 重新提所有历史任务训练集的特征、重建类均值。这
# 不是梯度意义上的 replay，但**在评估期访问了历史训练数据**，而 R7 要求把这一点明确
# 拆开看：如果改成「每个任务到来时把原型存下来，此后不再碰旧数据」，精度会掉多少？
#
# 两者的差别只在**原型**，不在测试特征：
#   * 终态重放：proto_j = mean(θ_T 特征)，测试特征也是 θ_T 的 —— 两侧同分布；
#   * 严格序贯：proto_j = mean(θ_j 特征)，测试特征仍是 θ_T 的 —— **原型漂移**。
# 所以两者之差恰好是「原型与当前特征空间的失配」，这就是审稿人要的那个数。
#
# 实现上只做两件事：
#   1. `task_prototypes()`：在任务 t 结束时，**只读任务 t 的训练集**（严格序贯允许的
#      全部信息），算该任务 C 个类的原型，存成 ncm_proto_task{t}.pt；
#   2. `strict_final_eval()`：训练全部结束后，用终态特征对**存下来的**原型做分类。
# 注意不能复用 `ncm_row()`：它每次都要重读 0..t 全部训练集，那在严格协议下是不允许的，
# 而且会把「原型漂移」这个待测量本身抹掉。


@torch.no_grad()
def task_prototypes(model, split, task_id: int, classes_per_task: int,
                    device, batch_size: int = 64, num_workers: int = 0):
    """只用**当前任务**的训练集，算该类簇的原型。返回 (classes_per_task, D)。

    时间语义：在任务 task_id 训练完之后、`after_task` 之前调用，所以用的是 θ_t ——
    这正是「任务到来时存下原型」的那一版原型。
    """
    was_training = model.training
    model.eval()
    try:
        tr = split.task_dataset(task_id, "train")
        feats, labels = _extract(model, tr, device, batch_size, num_workers)
    finally:
        model.train(was_training)

    if feats.numel() == 0:
        raise RuntimeError(f"strict proto: task {task_id} 训练集为空")
    trn = F.normalize(feats, dim=1)
    lo = task_id * classes_per_task
    means = torch.zeros(classes_per_task, trn.shape[1])
    for i in range(classes_per_task):
        idx = (labels == lo + i).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise RuntimeError(
                f"strict proto: 任务 {task_id} 的类 {lo + i} 没有训练样本")
        means[i] = F.normalize(trn[idx].mean(dim=0), dim=0)
    return means


@torch.no_grad()
def strict_final_eval(model, split, proto_dir, classes_per_task: int, num_tasks: int,
                      device, batch_size: int = 64, num_workers: int = 0):
    """终态特征 × 各任务当场存下的原型。返回 (acc_cil, acc_til, per_task_cil)。

    与主表 acc_cil 的唯一差别是原型的来历（θ_j 而非 θ_T），其余口径逐字相同：
    同样的 L2 归一化、同样的余弦最近邻、同样的已见全类候选集。
    """
    import torch as _torch
    rows = []
    for t in range(num_tasks):
        p = proto_dir / f"ncm_proto_task{t}.pt"
        if not p.exists():
            # 缺一块就无法构造 100 类的判别器，宁可报错也别用重算的原型悄悄补上
            raise FileNotFoundError(f"strict proto: 缺 {p}（任务 {t} 的原型没存下来）")
        rows.append(_torch.load(p, map_location="cpu", weights_only=True))
    means = _torch.cat(rows, dim=0)                       # (num_tasks*C, D)

    was_training = model.training
    model.eval()
    try:
        te = ConcatDataset([split.task_dataset(j, "test") for j in range(num_tasks)])
        te_feats, te_labels = _extract(model, te, device, batch_size, num_workers)
    finally:
        model.train(was_training)
    ten = F.normalize(te_feats, dim=1)

    per_task = [0.0] * num_tasks
    per_task_til = [0.0] * num_tasks
    for j in range(num_tasks):
        lo, hi = j * classes_per_task, (j + 1) * classes_per_task
        idx = (te_labels >= lo) & (te_labels < hi)
        if not bool(idx.any()):
            continue
        x, y = ten[idx], te_labels[idx]
        per_task[j] = (x @ means.t()).argmax(dim=1).eq(y).float().mean().item()
        pred = (x @ means[lo:hi].t()).argmax(dim=1) + lo
        per_task_til[j] = pred.eq(y).float().mean().item()
    return (sum(per_task) / num_tasks,
            sum(per_task_til) / num_tasks,
            per_task)
