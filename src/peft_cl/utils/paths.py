"""运行目录规划：一次实验的产物统一放在一个确定性目录下（断点续训 + 网格调度用）。"""

from pathlib import Path


def run_dir(config) -> Path:
    """根据配置推导实验目录：experiments/{benchmark}/{method}/{label}/seed{seed}/。"""
    label = config.tag or "default"
    return Path(config.out_dir) / config.benchmark / config.method / label / f"seed{config.seed}"
