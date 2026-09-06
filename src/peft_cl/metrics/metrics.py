"""持续学习评估指标。

准确率矩阵 acc_matrix[t][j] 表示「学完任务 t 之后，在任务 j 的测试集上的准确率」。
类增量（CIL）用全头 argmax（模型必须在所有已见类里选），任务增量（TIL）用
任务内 argmax（假定已知任务 id）。两个矩阵都记录，主指标用 CIL。
"""

from typing import List, Optional, Tuple

import torch


@torch.no_grad()
def evaluate(model, loader, device, class_range: Optional[Tuple[int, int]] = None) -> float:
    """在 loader 上算准确率。

    class_range=(lo, hi) 时只在 [lo, hi) 内取 argmax（TIL），否则全头 argmax（CIL）。
    """
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if class_range is not None:
            lo, hi = class_range
            pred = logits[:, lo:hi].argmax(dim=1) + lo
        else:
            pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def final_average_accuracy(acc_matrix: List[List[float]]) -> float:
    """最终平均准确率：最后一行的均值（学完全部任务后，各任务准确率的平均）。"""
    return float(torch.tensor(acc_matrix[-1]).float().mean())


def forgetting(acc_matrix: List[List[float]]) -> float:
    """平均遗忘（backward transfer）：每个任务从「历史峰值」到最后掉点的平均。"""
    T = len(acc_matrix)
    acc = torch.tensor(acc_matrix, dtype=torch.float)
    drops = []
    for j in range(T):
        peak = max(float(acc[t][j]) for t in range(j, T))
        drops.append(peak - float(acc[-1][j]))
    return float(torch.tensor(drops).mean())


def average_incremental_accuracy(acc_matrix: List[List[float]]) -> float:
    """平均增量准确率：所有 (t, j≤t) 的准确率均值，衡量整个学习轨迹的整体表现。"""
    T = len(acc_matrix)
    acc = torch.tensor(acc_matrix, dtype=torch.float)
    vals = [float(acc[t][j]) for t in range(T) for j in range(t + 1)]
    return float(torch.tensor(vals).mean())
