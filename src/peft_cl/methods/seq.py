"""Seq-LoRA：朴素串行微调基线（无任何抗遗忘保护，作为下界）。"""

from ..adapters.lora import inject_lora
from .base import CLMethod


class SeqMethod(CLMethod):
    """每个任务直接在共享 LoRA 上微调，无正则、无重要性 —— 灾难性遗忘最严重。"""

    name = "seq"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.loras = inject_lora(model, config.lora_rank, config.lora_alpha)

    def regularization_loss(self):
        return 0.0
