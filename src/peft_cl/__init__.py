"""peft-cl：参数高效微调（PEFT）在持续学习场景下的灾难性遗忘研究。

本仓库实现并对比多种「低秩适配（LoRA）+ 持续学习（Continual Learning）」方法，
核心贡献为 FOLoRA（Fisher-Orthogonalized Low-Rank Adaptation）。

子模块：
- data      数据集加载 + 类增量切分
- backbone  冻结的 ViT 主干 + 可扩展分类头
- adapters  LoRA 适配器
- fisher    经验 Fisher 估计（对角 + 低秩核）
- methods   seq / ewc / olora / folora 各方法
- trainer   持续学习训练循环（含断点续训）
- metrics   平均准确率 / 遗忘 / 前向迁移
"""

__version__ = "0.1.0"
