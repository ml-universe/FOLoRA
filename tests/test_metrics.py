"""评估指标单元测试（纯 CPU）。"""

from peft_cl.metrics.metrics import (average_incremental_accuracy,
                                     final_average_accuracy, forgetting)


def test_final_average_accuracy():
    acc = [[0.5, 0.0], [0.6, 0.8]]
    assert abs(final_average_accuracy(acc) - 0.7) < 1e-6


def test_forgetting_zero_when_no_drop():
    acc = [[0.5, 0.0], [0.5, 0.8]]
    assert abs(forgetting(acc) - 0.0) < 1e-6


def test_forgetting_measures_drop():
    # 任务 0 从峰值 0.9 掉到 0.3（忘 0.6），任务 1 无遗忘 -> 平均遗忘 0.3
    acc = [[0.9, 0.0], [0.3, 0.8]]
    assert abs(forgetting(acc) - 0.3) < 1e-6


def test_average_incremental_accuracy():
    acc = [[0.5, 0.0], [0.6, 0.8]]
    assert abs(average_incremental_accuracy(acc) - (0.5 + 0.6 + 0.8) / 3) < 1e-6
