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

    同时关闭 cudnn 自动算法搜索（benchmark=False）并启用确定性卷积，
    保证同 seed 下结果可复现（代价是轻微变慢，可接受）。
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
