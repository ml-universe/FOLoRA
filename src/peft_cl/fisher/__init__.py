"""Fisher 估计：经验对角 Fisher（EWC 用）+ 低秩输出核（FOLoRA 用）。"""
from .fisher import diagonal_fisher, output_kernel_fisher

__all__ = ["diagonal_fisher", "output_kernel_fisher"]
