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

    def before_task(self, task_id: int, train_loader=None) -> None:
        """任务开始前调用。

        train_loader 为该任务的训练集 DataLoader（可能为 None）。InfLoRA 需要用它
        在「学习新任务之前」收集新任务的输入空间来设计 B_t，故必须传。
        """

    def after_task(self, task_id: int, train_loader, device) -> None:
        """任务训练结束后调用，估计/累积重要性。"""

    def regularization_loss(self):
        """返回正则损失（标量，基类为 0.0；子类重写返回 torch 张量）。"""
        return 0.0

    def rebuild_for_resume(self, last_task_id: int) -> None:
        """断点续训时重建「逐任务动态创建」的结构（O-LoRA / InfLoRA 的 per-task adapter）。

        必须在 trainer 把 checkpoint 的 model_state 灌进模型**之前**调用：
        那些 `...adapters.N.lora_A.weight` 的键只有在模型里已存在对应参数时才能被加载，
        否则会被 load_state_dict(strict=False) 静默丢弃 —— 表现为「续训后旧任务的
        adapter 全变成随机初始化」，结果错得很隐蔽。子类按需重写。
        """

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass


def lora_list(model):
    """返回模型里所有单 LoRA 适配器（seq/ewc/folora 用）。"""
    return list(iter_lora(model))
