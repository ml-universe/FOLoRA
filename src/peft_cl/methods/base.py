"""持续学习方法基类。

约定（训练器按此调用）：
- __init__      注入适配器、初始化方法状态；
- before_task   任务训练前准备（如 O-LoRA 加 adapter、FOLoRA 分解保护子空间）；
- after_task    任务训练后估计并累积重要性（EWC 的 Fisher、FOLoRA 的核）；
- regularization_loss  返回要加到交叉熵上的正则损失标量；
- state_dict / load_state_dict  方法状态的序列化（断点续训必需，张量统一放 CPU）。
"""

from ..adapters.lora import iter_lora


class CLMethod:
    name = "base"

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def before_task(self, task_id: int) -> None:
        """任务开始前调用。"""

    def after_task(self, task_id: int, train_loader, device) -> None:
        """任务训练结束后调用，估计/累积重要性。"""

    def regularization_loss(self):
        """返回正则损失（标量，基类为 0.0；子类重写返回 torch 张量）。"""
        return 0.0

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass


def lora_list(model):
    """返回模型里所有单 LoRA 适配器（seq/ewc/folora 用）。"""
    return list(iter_lora(model))
