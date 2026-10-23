"""评估指标：准确率、遗忘、平均增量准确率。"""
from .metrics import (average_incremental_accuracy, evaluate, final_average_accuracy,
                      forgetting)

__all__ = ["evaluate", "final_average_accuracy", "forgetting",
           "average_incremental_accuracy"]
