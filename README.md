# FOLoRA: Fisher-Orthogonalized Low-Rank Adaptation for Continual Learning

参数高效微调（PEFT）+ 持续学习（Continual Learning）场景下**灾难性遗忘**的研究项目。
核心方法 **FOLoRA** 用二阶 Fisher 信息在低秩子空间内做「重要性加权的软正交投影」：
重要方向（高 Fisher）强保护防遗忘，次要方向允许复用以保塑性。目标投稿 Neurocomputing（CCF-C）。

## 环境

- GPU：RTX 5060 Laptop **8GB**（Blackwell sm_120），需 torch CUDA 版
- Python 3.11+，Windows
- 网络：HuggingFace 被墙 → 模型走 download.pytorch.org / 数据走 ModelScope

```bash
# 1) 安装 CUDA 版 torch（必须用 cu128 索引，勿用 pip 默认源）
pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128

# 2) 可编辑安装本项目（使 `from peft_cl import ...` 可用）
pip install -e .

# 3) 下载 CIFAR-100（走 ModelScope 镜像，cs.toronto.edu 国内极慢）
python -m scripts.download_cifar
```

## 快速开始

```bash
# 冒烟测试（小规模验证全流程正确性）
python -m scripts.run_single --benchmark cifar100 --num_tasks 5 --method folora_v2 --epochs 1 --lora_rank 4 --tag smoke

# 单次实验（20 任务 × 5 类，类增量）
python -m scripts.run_single --benchmark cifar100 --num_tasks 20 --method folora_v2 --seed 0

# 全量队列（seed 对齐 → EWC 调优 → prompt 基线 → 汇总；幂等 + 断点续训，重跑自动跳过已完成）
python -m scripts.run_full_queue

# 断电续训：直接重新运行即可（各脚本均幂等，自动从 checkpoint 续跑）

# 汇总结果 → reports/summary_cifar100.md
python -m scripts.aggregate --benchmark cifar100
```

## 方法

| method | 说明 |
|---|---|
| `seq`   | 朴素串行 LoRA（无保护，下界） |
| `ewc`   | EWC-LoRA（对角 Fisher 惩罚 LoRA 参数） |
| `olora` | O-LoRA（每任务独立 adapter + 正交初始化） |
| `l2p`   | L2P（prompt 池 + 余弦 top-k 选择 + key loss） |
| `coda`  | CODA-Prompt（prompt 组件 + 注意力软加权） |
| `folora_v2`| **FOLoRA（参数 Fisher 加权的软正交投影，本文方法，λ=300/k=16）** |

## 关键超参

- `--lora_rank` LoRA 秩（默认 16）
- `--folora_lambda` FOLoRA 正交正则强度 λ（默认 300）
- `--folora_topk` 保护方向数 top-k（0 = 取满 r 个方向）
- `--fisher_batches` Fisher 估计样本数（逐样本梯度，默认 300）

## 断点续训机制

每训练完一个任务立即把「模型可训参数 + 方法状态（累积 Fisher 核）+ RNG 状态 + 类顺序
+ 准确率矩阵」落盘为 `checkpoint.pt`。每个任务约 1-2 分钟，断电最多损失当前任务。
新进程 `python -m scripts.run_single --resume`（或 run_grid）自动检测 checkpoint 并从
「上一个已完成任务 + 1」续跑，结果与一气呵成完全一致。全部完成写 `results.json`
（含 `finished: true`），网格调度器据此跳过已完成实验。

## 目录结构

```
src/peft_cl/
  data/      数据集加载 + 类增量切分
  backbone/  冻结 ViT + 可扩展分类头
  adapters/  LoRA / 多任务 LoRA
  fisher/    逐样本 Fisher 估计（对角 + 低秩核）
  methods/   seq / ewc / olora / folora
  trainer/   持续学习训练循环（断点续训）
  metrics/   平均准确率 / 遗忘 / 前向迁移
scripts/     下载、单次运行、网格调度、汇总
paper/       LaTeX 论文（方法 + 理论推导 + 实验）
tests/       pytest 单元测试（CPU）
```

## 测试

```bash
python -m pytest tests/ -q
```
