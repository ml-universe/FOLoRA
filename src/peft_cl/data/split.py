"""类增量切分：把 C 类随机打乱后切成 num_tasks 个任务，每个任务 k=C/num_tasks 类。

核心对象 ContinualSplit 负责三件事：
1. 类顺序 class_order —— 固定 seed 可复现、可保存（断点续训必需）；
2. 原始类别标签 → 全局连续索引的映射（供可扩展分类头使用，输出维度 = 已见类别数）；
3. 按任务取数据子集（用 .targets 快速过滤标签，不触发图像 transform）。
"""

import random
from typing import List

from torch.utils.data import Dataset


def make_class_order(num_classes: int, num_tasks: int, seed: int) -> List[List[int]]:
    """生成类顺序：把 [0..C-1] 用 seed 洗牌后切成 num_tasks 个任务。

    返回 list[list[int]]：外层是任务、内层是该任务的「全局类索引」（0..C-1）。
    """
    assert num_classes % num_tasks == 0, "类总数必须能被任务数整除"
    rng = random.Random(seed)
    order = list(range(num_classes))
    rng.shuffle(order)
    k = num_classes // num_tasks
    return [order[i * k:(i + 1) * k] for i in range(num_tasks)]


class RemappedSubset(Dataset):
    """取 base 的子集，并把原始类别标签重映射为「全局连续索引」。"""

    def __init__(self, base, indices, class_to_global):
        self.base = base
        self.indices = indices
        self.class_to_global = class_to_global

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.class_to_global[int(y)]


class ContinualSplit:
    """把一个基础数据集（含 .targets 原始标签）按类增量切分成多任务。"""

    def __init__(self, base_train, base_test, class_order: List[List[int]]):
        self.base_train = base_train
        self.base_test = base_test
        self.class_order = class_order
        self.num_tasks = len(class_order)
        # 原始类别标签 -> 全局连续索引（class_order 展平序列中的位置 0..C-1）
        self.class_to_global = {}
        for gi, cls in enumerate([c for task in class_order for c in task]):
            self.class_to_global[cls] = gi
        self.num_classes = len(self.class_to_global)

    def set_class_order(self, class_order: List[List[int]]) -> None:
        """用保存的类顺序重建映射（断点续训恢复用）。"""
        self.class_order = class_order
        self.num_tasks = len(class_order)
        self.class_to_global = {}
        for gi, cls in enumerate([c for task in class_order for c in task]):
            self.class_to_global[cls] = gi
        self.num_classes = len(self.class_to_global)

    def task_dataset(self, task_id: int, split: str) -> RemappedSubset:
        """返回指定任务、指定 split（train/test）的数据子集，标签已重映射为全局索引。"""
        base = self.base_train if split == "train" else self.base_test
        gids = {self.class_to_global[c] for c in self.class_order[task_id]}
        targets = base.targets
        indices = [i for i, y in enumerate(targets)
                   if self.class_to_global.get(int(y), -1) in gids]
        return RemappedSubset(base, indices, self.class_to_global)

    def task_classes(self, task_id: int) -> List[int]:
        """返回该任务的全局索引列表（评估时按类统计准确率用）。"""
        return [self.class_to_global[c] for c in self.class_order[task_id]]
