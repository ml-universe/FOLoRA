"""随机种子工具。

为什么单独一个模块：持续学习实验对随机性高度敏感（类顺序、数据增强、初始化），
必须三级（python/numpy/torch）同时固定才能复现；而断点续训要求 RNG 状态能被
「保存 + 恢复」，否则续训与一气呵成跑出的结果对不上。
"""

import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定 python / numpy / torch / cuda 的随机种子。

    同时关闭 cudnn 自动算法搜索（benchmark=False）并启用确定性卷积。

    ⚠️ 2026-10-03 实测更正：原先这里写着「**保证同 seed 下结果可复现**」，
    那是**假的**。同 seed、同代码、同超参连跑两次（O-LoRA / CIFAR-100 / seed 0），
    逐 epoch loss 从 **task 0 的第一个 epoch** 起就分叉，12 个 epoch 无一相同。
    原因：本函数只设了 `cudnn.deterministic`，**没有**调
    `torch.use_deterministic_algorithms(True)`；AMP/GradScaler 与若干含原子加
    的 backward 算子在 torch 的默认模式下属「与实现相关的非确定性」，不受
    `cudnn.deterministic` 约束。cudnn.deterministic 只覆盖卷积算法选择那一层。

    后果（不要据此推翻结论）：**每次重训都是一次新的抽样**，不能用
    `torch.equal` 之类的逐位比较去断言「某处代码改动只影响存盘、不影响训练」。
    跨 seed 的 mean±std 仍然有效（run-to-run 抖动已混在里面，只会让 std 偏大、
    不会偏小），但**单个 seed 的数值不可复现**。

    若要真正逐位可复现，需另加 `torch.use_deterministic_algorithms(True)`
    （可能对尚无确定性实现的算子直接报错，且会改变既有全部历史数字的可比性，
    故**不可**顺手打开，须作为一次独立决定）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state() -> Dict[str, Any]:
    """导出当前全部 RNG 状态，供 checkpoint 保存。

    返回 dict 可直接 torch.save（python 的 getstate 是普通对象，可 pickle）。
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    """恢复 RNG 状态（断点续训时调用，保证与不中断结果一致）。"""
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
