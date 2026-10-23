"""类增量切分逻辑的单元测试（纯 CPU）。"""

from peft_cl.data.split import ContinualSplit, make_class_order


class _DummyDataset:
    """最小数据集：只提供 .targets 与 __getitem__，供切分测试用。"""

    def __init__(self, labels):
        self.targets = list(labels)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return i, self.targets[i]


def test_make_class_order_partition_and_determinism():
    o1 = make_class_order(100, 20, 42)
    o2 = make_class_order(100, 20, 42)
    assert o1 == o2, "同 seed 类顺序应确定"
    flat = [c for task in o1 for c in task]
    assert sorted(flat) == list(range(100)), "类顺序应覆盖 0..99 无重复"
    assert all(len(t) == 5 for t in o1), "20 任务 × 5 类"


def test_continual_split_task_dataset_and_remap():
    train = _DummyDataset(range(10))
    test = _DummyDataset(range(10))
    order = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]  # 5 任务 × 2 类
    split = ContinualSplit(train, test, order)
    assert split.num_classes == 10 and split.num_tasks == 5

    # 任务 1 是原始类 2,3 -> 全局索引 2,3
    d = split.task_dataset(1, "train")
    assert len(d) == 2
    labels = {d[i][1] for i in range(len(d))}
    assert labels == {2, 3}
    assert split.task_classes(1) == [2, 3]


def test_set_class_order_rebuilds_mapping():
    train = _DummyDataset(range(10))
    test = _DummyDataset(range(10))
    split = ContinualSplit(train, test, [[0, 1], [2, 3]])
    split.set_class_order([[9, 8], [7, 6]])
    assert split.class_to_global[9] == 0 and split.class_to_global[6] == 3
