"""界的实证探针：在任务边界测「二阶主项 vs 实测遗忘」，并定出三阶余项的大小。

为什么需要它
------------
`paper/sections/04_theory.tex` 有两处承诺要在 Sec. discussion 里做实证：

  1. 「We verify empirically that the higher-order term is negligible relative to
     the quadratic term in Sec. discussion.」
  2. 「...quantify it empirically in Sec. discussion.」（指 Assumption 1 的违背）

实际 `sec:discussion` 里两条都没有——承诺了不做比不说更伤。本模块把这两个数真的
测出来。

为什么必须在训练过程里测，而不能对已完成 run 的 checkpoint 事后测
------------------------------------------------------------------
`after_task` 会把 `ref_params` 覆盖成**当前**任务的参数快照，所以训练结束后落盘的
checkpoint 里 `ref_params == model_state`（已实测：‖cur-ref‖ = 0）。也就是说事后从
单个 checkpoint 拿不到位移 δθ——它是零。真正可用的三元组

    (θ_{t-1}, θ_t, F̄_{<t})

只在 `after_task` **执行之前**同时存在于内存中：此刻 `ref_params` 还停在上一个任务
的边界，`acc_grads` 还没并入本任务的梯度，而模型已经是 θ_t。因此探针挂在
`trainer._train_task` 里 `_evaluate` 之后、`after_task` 之前。

方法：α 插值，而不是去估 L_3
----------------------------
余项是 (L_3/6)‖δθ‖³，而 L_3（Hessian 的 Lipschitz 常数）无法从任何 checkpoint 测出，
所以「余项可忽略」不能靠算 L_3 来证。可行的做法是直接测**非线性本身**：

沿真实位移方向插值 θ(α) = θ_{t-1} + α·δθ，测前面所有任务的总损失
    dL(α) = Σ_{τ<t} [ L_τ(θ(α)) - L_τ(θ_{t-1}) ]
再拟合 dL(α) = a·α + b·α² + c·α³。于是

- a：Assumption 1（θ_{t-1} 处梯度为零）沿该方向的残留。a≈0 说明线性项确实小。
- b：界里二阶主项的**实测**系数，可直接与 ½δθᵀF̄δθ 对比——这是对 Fisher 估计
  「是否标定了真实曲率」的检验，比「余项小」更强的一句。
- c：三阶项。|c|/|b| 就是「高阶项相对二阶项」的直接度量；≤0.1 则承诺 1 成立。

这测的是**真实位移尺度上**的非线性，而不是 α→∞ 的渐近行为，正好是定理关心的那个
尺度，也因此不需要知道 L_3。

适用范围（要写进论文，不要含糊）
--------------------------------
- **head 固定**：只插值 adapter，分类头不动。这正好等于定理的作用域——04_theory.tex
  明说界只覆盖 adapter（「we do not model its contribution」），且在 NCM 协议下分类头
  本来就被丢弃。head 与 α 无关，只影响 dL 的基线常数，不影响曲线形状。
- 只适用于维护 `acc_grads`/`ref_params` 的方法（FOLoRA 系）。

本模块**绝不允许把训练搞崩**：所有异常在调用处被吞掉并记日志，见 trainer 的调用点。
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def flatten_adapter(lora) -> torch.Tensor:
    """与 FOLoRAv2Method._flatten 完全同序：A 在前、B 在后。

    顺序必须一致，否则和 `ref_params` 对不上——那样算出来的 δθ 是乱的，而且**不会
    报错**，只会给出一个看似合理的错误数字。
    """
    return torch.cat([lora.lora_A.weight.reshape(-1),
                      lora.lora_B.weight.reshape(-1)])


def unflatten_into(lora, flat: torch.Tensor) -> None:
    with torch.no_grad():
        n_a = lora.lora_A.weight.numel()
        lora.lora_A.weight.copy_(flat[:n_a].reshape(lora.lora_A.weight.shape))
        lora.lora_B.weight.copy_(flat[n_a:].reshape(lora.lora_B.weight.shape))


@torch.no_grad()
def _loss_sum(model, split, task_ids, device, batch_size, max_per_task=0) -> float:
    """Σ_{τ∈task_ids} 该任务测试集上的 CE 损失（head 固定 -> 只反映 adapter 的位移）。

    用测试集而非训练集：训练集损失被记忆效应压低，测不出遗忘。
    """
    total, n = 0.0, 0
    for t in task_ids:
        ds = split.task_dataset(t, "test")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        seen = 0
        for x, y in loader:
            if max_per_task and seen >= max_per_task:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += torch.nn.functional.cross_entropy(
                logits, y, reduction="sum").item()
            n += y.numel()
            seen += y.numel()
    return total / max(n, 1)


def probe_bound_terms(model, method, split, device, task_id: int,
                      alphas=ALPHAS, batch_size: int = 128,
                      max_per_task: int = 0) -> dict | None:
    """测一个任务边界的界各项。返回 None 表示该方法/该边界不适用。"""
    loras = getattr(method, "loras", None)
    if not loras or not getattr(method, "acc_grads", None) \
            or not getattr(method, "ref_params", None):
        return None
    if task_id < 1:
        return None                       # 没有「之前的任务」，界的 LHS 为空

    prev_tasks = list(range(task_id))
    cur = [flatten_adapter(l).detach().float().clone() for l in loras]
    prv = []
    for i, lora in enumerate(loras):
        if i not in method.ref_params:
            return None                   # ref_params 不完整时宁可不测，也不要测错
        prv.append(method.ref_params[i].float().reshape(-1).clone())
    delta = [c - p for c, p in zip(cur, prv)]

    # 界的二阶主项 ½δθᵀ F̄ δθ，与 regularization_loss 同式（只差系数 1/2）。
    # 此刻 acc_grads 还没并入本任务 -> 恰好是 F̄_{<t}，与训练本任务时正则项用的
    # 就是同一个矩阵（这正是必须在 after_task 之前测的原因）。
    quad = 0.0
    for i, _ in enumerate(loras):
        if i not in method.acc_grads:
            continue
        # **必须搬到 delta 所在设备**：`acc_grads` 按设计**常驻 CPU**（见 folora_v2.py 的
        # 注释：为了 state_dict 序列化），而 delta 由 CUDA 上的 lora 权重与 ref_params 相减
        # 得到。漏掉这一步会在 CUDA 上报 `Expected all tensors to be on the same device,
        # but got mat is on cpu ... wrapper_CUDA_addmv_` —— 而单测是**纯 CPU** 的，
        # 所以这条路径在 2026-09-25 之前从未被任何测试覆盖，探针连续 19 个边界全败、
        # 零成功测量。方法自身在 folora_v2.py 里就是这么处理的（`.to(dtheta.device)`），
        # 这里照抄同一模式。
        G = method.acc_grads[i].float().to(delta[i].device)
        quad += float((G @ delta[i]).pow(2).sum())
    quad *= 0.5

    was_training = model.training
    model.eval()
    try:
        curve = []
        for alpha in alphas:
            for i, lora in enumerate(loras):
                unflatten_into(lora, prv[i] + alpha * delta[i])
            curve.append((float(alpha),
                          _loss_sum(model, split, prev_tasks, device,
                                    batch_size, max_per_task)))
        # 无论后面发生什么，adapter 必须还原成 θ_t
        for i, lora in enumerate(loras):
            unflatten_into(lora, cur[i])
    finally:
        model.train(was_training)

    dL = np.array([L - curve[0][1] for _, L in curve], dtype=np.float64)
    A = np.stack([np.array(alphas, dtype=np.float64) ** p for p in (1, 2, 3)], axis=1)
    coef, *_ = np.linalg.lstsq(A, dL, rcond=None)
    a, b, c = (float(x) for x in coef)

    A2 = A[:, :2]
    pred2 = A2 @ np.linalg.lstsq(A2, dL, rcond=None)[0]
    ss_tot = float(((dL - dL.mean()) ** 2).sum())
    r2_quad = 1.0 - float(((dL - pred2) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    return {
        "task_id": task_id,
        "n_prev_tasks": len(prev_tasks),
        "delta_norm": float(torch.cat(delta).norm()),
        "quad_term": quad,                       # ½δθᵀ F̄_{<t} δθ
        "fit_a_linear": a,
        "fit_b_quadratic": b,
        "fit_c_cubic": c,
        "linear_over_quad": (a / b) if b else float("nan"),
        "cubic_over_quad": (c / b) if b else float("nan"),
        "fisher_over_fitted": (quad / b) if b else float("nan"),
        "r2_quadratic_only": r2_quad,
        "dL_at_alpha1": float(dL[list(alphas).index(1.0)]) if 1.0 in tuple(alphas) else None,
        "dL_curve": dL.tolist(),
        "alphas": [float(x) for x in alphas],
    }


def append_jsonl(path: Path, record: dict) -> None:
    """追加一行 JSON。训练过程里写失败也不能影响训练，所以调用处吞异常。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
