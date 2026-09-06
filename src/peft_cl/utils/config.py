"""实验配置。

为什么用 dataclass 而不是 YAML：需要「可 JSON 序列化」以便写进 checkpoint，
续训时直接反序列化恢复同一份配置，避免「改脚本默认值导致续训参数漂移」的坑。
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class CLConfig:
    """一次持续学习实验的完整配置。字段注释即默认值的语义。"""

    # ---- 数据 ----
    benchmark: str = "cifar100"        # cifar10 / cifar100 / tinyimagenet
    num_tasks: int = 20                # 任务数（类总数须能被 num_tasks 整除）
    data_root: str = "data"            # 数据缓存目录
    image_size: int = 224              # 输入分辨率（ViT 需 224）

    # ---- 模型 ----
    backbone: str = "vit_b_16"         # 主干（当前仅 vit_b_16）
    lora_rank: int = 16                # LoRA 秩 r
    lora_alpha: int = 16               # LoRA 缩放 α（scale = α/r）

    # ---- 训练 ----
    method: str = "folora"             # seq / ewc / olora / folora
    seed: int = 0
    epochs: int = 5                    # 每任务训练轮数
    batch_size: int = 32               # 8GB 卡 224px 用 32（64 会顶满显存触发共享内存交换）
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0               # Windows 下 0 最稳（避免 spawn 递归）
    use_amp: bool = True               # bf16 混合精度（tensor core 提速 + 省显存）

    # ---- 方法超参 ----
    fisher_batches: int = 300          # Fisher 估计的「样本数」（逐样本梯度，EWC/FOLoRA 共用）
    folora_lambda: float = 300.0       # FOLoRA 正交正则强度 λ（v2 扫描定最优，见 reports/v2_full_sweep.log）
    folora_topk: int = 0               # 保护方向数 top-k（0 = 取满 r 个方向 = rank）
    ewc_lambda: float = 100.0          # EWC 正则强度 λ

    # ---- prompt 基线超参（L2P / CODA-Prompt）----
    prompt_pool_size: int = 20         # L2P prompt 池大小 / CODA 组件数
    prompt_length: int = 5             # 单个 prompt 的 token 数
    prompt_topk: int = 5               # L2P 每输入选择的 top-k prompt 数

    # ---- 运行 ----
    device: str = "cuda"
    out_dir: str = "experiments"       # 结果根目录
    eval_every: int = 1                # 每训练完一个任务评估一次（固定 1）
    tag: str = ""                      # 实验标签（区分消融，如 lam1.0_topk16）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CLConfig":
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CLConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
