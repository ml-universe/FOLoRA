"""EWC-LoRA 基线：对角 Fisher 重要性加权的二次惩罚，作用在 LoRA 参数上。

只正则化 LoRA 参数（不含分类头），保证与 FOLoRA 公平对比——差异只在「如何保护
适配器」，分类头双方都只用交叉熵。
"""

from ..adapters.lora import inject_lora
from ..fisher.fisher import diagonal_fisher
from .base import CLMethod


class EWCMethod(CLMethod):
    name = "ewc"

    def __init__(self, model, config):
        super().__init__(model, config)
        inject_lora(model, config.lora_rank, config.lora_alpha)
        self.ref_params = {}   # 参数名 -> 上一任务结束后的参数快照（device）
        self.fisher = {}       # 参数名 -> 累积对角 Fisher（device）

    def _adapter_param_names(self):
        """只取 LoRA 参数名（含 'lora_A'/'lora_B'），排除分类头。"""
        return [n for n, p in self.model.named_parameters()
                if p.requires_grad and ("lora_A" in n or "lora_B" in n)]

    def after_task(self, task_id, train_loader, device):
        f_new = diagonal_fisher(self.model, train_loader, self.config.fisher_batches, device)
        for name in self._adapter_param_names():
            if name in f_new:
                f = f_new[name].detach().to(device)
                self.fisher[name] = self.fisher.get(name, 0.0) + f
        self.ref_params = {
            name: dict(self.model.named_parameters())[name].detach().clone()
            for name in self._adapter_param_names()
        }

    def regularization_loss(self):
        if not self.ref_params:
            return 0.0
        named = dict(self.model.named_parameters())
        loss = 0.0
        for name, ref in self.ref_params.items():
            p = named[name]
            loss = loss + (self.fisher[name] * (p - ref) ** 2).sum()
        return self.config.ewc_lambda * loss

    def state_dict(self):
        return {
            "ref_params": {k: v.cpu() for k, v in self.ref_params.items()},
            "fisher": {k: v.cpu() for k, v in self.fisher.items()},
        }

    def load_state_dict(self, d):
        self.ref_params = {k: v for k, v in d["ref_params"].items()}
        self.fisher = {k: v for k, v in d["fisher"].items()}

    def before_task(self, task_id):
        # 恢复时张量可能在 CPU，移到当前设备
        dev = self.config.device
        self.ref_params = {k: v.to(dev) for k, v in self.ref_params.items()}
        self.fisher = {k: v.to(dev) for k, v in self.fisher.items()}
