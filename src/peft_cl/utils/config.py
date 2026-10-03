"""实验配置。

为什么用 dataclass 而不是 YAML：需要「可 JSON 序列化」以便写进 checkpoint，
续训时直接反序列化恢复同一份配置，避免「改脚本默认值导致续训参数漂移」的坑。
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .io import atomic_write_json


@dataclass
class CLConfig:
    """一次持续学习实验的完整配置。字段注释即默认值的语义。"""

    # ---- 数据 ----
    benchmark: str = "cifar100"        # cifar100（论文主用）/ imagenetr（论文次用）/ cifar10
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
    folora_lambda: float = 300.0       # FOLoRA 正交正则强度 λ。注意：这是 v2 全网格扫描期的
                                       # 旧默认；**论文主配置是 λ=3**（tab:ablation 的预注册
                                       # 主配置）。复现论文须显式 --folora_lambda 3。
    folora_topk: int = 0               # 保护方向数 top-k（0 = 取满 r 个方向 = rank）。
                                       # 同上：**论文主配置是 k=64**（取满秩是扫描期旧默认）。
    folora_weighted: bool = True       # True=按 Fisher 特征值 σ² 加权；False=各方向等权（消融用，见 folora_v2 注释）
    ewc_lambda: float = 100.0          # EWC 正则强度 λ
    # O-LoRA 推理聚合方式。sum = 论文原式 W0 + Σ_i B_i A_i；mean = 除以任务数（公平性对照）。
    # 原式下 ΔW 的秩上限随任务数线性增长（rank 16 × 20 任务 = 320 / 768 ≈ 42%），
    # 会把 NCM 特征推歪，实测 64.79（低于 Seq-LoRA 的 69.31）。mean 是「若换个聚合方式
    # 能不能救回来」的对照——两者都报，把「你复现坏了」这个质疑用数据挡掉。
    olora_aggregate: str = "sum"
    # O-LoRA 的正交性约束强度 λ₁（原文目标 = Σlog p + λ₁ Σ_{i<t} ‖A_tᵀA_i‖_F²）。
    # **默认 0.0 = 只做正交初始化、不施加训练期约束**——这正是已完成的那 10 个 run 的
    # 语义，默认值保持 0.0 是为了让已有结果逐位可复现。文献建议 λ₁ 起手 0.1，先验掉得多
    # 就往 0.5~1.0 加。见 methods/olora.py 的说明。
    olora_orth_lambda: float = 0.0

    # ---- 诊断（默认全关，不影响主实验）----
    # 打开后在每个任务边界（_evaluate 之后、after_task 之前）测一次「界的二阶项 vs
    # 实测遗忘」的 α 插值曲线，写入 run 目录的 bound_terms.jsonl。默认 False：只有
    # 专门为了验证 04_theory.tex 的高阶项承诺才需要，主队列不要开（每个边界多几分钟）。
    # 跑法：scripts/run_lambda_grid.py --batch bound（supervise_all 的阶段 3i）。
    # 出数后**不要**用 scripts/bound_terms.py 汇总——那个脚本从 checkpoint 事后重算是
    # 无效的（ref_params 已被 after_task 覆盖成当前参数，实测 ‖cur-ref‖ = 0.000000）。
    # 用 scripts/bound_probe_report.py 读 run 目录下的 bound_terms.jsonl。
    bound_probe: bool = False

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
        # 原子写：config.json 每个任务都会重写，而 NCM 评估 / 资源统计都会读它。
        # 截断的 config.json 会让 json.load 抛异常，直接崩掉整个评估队列。
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> "CLConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
