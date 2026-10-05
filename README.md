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
#    含 pyarrow（CIFAR parquet 数据路径硬依赖）与 scipy（显著性检验）
pip install -e .            # 另需跑单元测试时：pip install -e ".[dev]"（额外装 pytest）

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

# 汇总结果（论文主表口径 = frozen-feature NCM）→ reports/ncm_summary_<bench>.json
python -m scripts.eval_ncm_sweep --benchmark cifar100
python -m scripts.eval_ncm_sweep --benchmark imagenetr

# 由 JSON 生成论文三张表（勿手改 paper/tables/*.tex）
python -m scripts.make_paper_tables

# 显著性检验 → reports/significance_*.md（论文 Table 1/2 的对照关系）
python -m scripts.significance --benchmark all
```

> ⚠️ `scripts/aggregate.py` 是**按协议分组**的诊断性汇总器：主指标同样从 `reports/ncm/`
> 读 NCM（旧 head-CIL 只作名字带 `LEGACY` 的列保留），它按每个 run 自己的 `config.json`
> 校验 20/5/16 主协议并输出 `reports/summary_<bench>.{json,md}`。
> **它与论文表格不是同一条产线**：论文的 Table 1/2/3 由 `make_paper_tables.py` 从
> `reports/ncm_summary_<bench>.json` 生成。两者口径一致但汇总方式与文件不同，
> 引用数字请以 `paper/tables/*.tex` 为准，不要把 `summary_*.md` 直接抄进论文。

## ⚠️ 两个同名异义的 `final_acc_cil`

复现论文数字前必须知道：

- `experiments/*/results.json` 的 `final_acc_cil` = **线性分类头的 CIL 准确率**（旧口径，量级 ~0.07）
- `reports/ncm/*.json` 的 `final_acc_cil` = **冻结特征 + 最近类均值（NCM）**准确率（量级 ~0.76）
  —— **论文正文与 Table 1 全部引用的是这一口径**

两者字段同名、量级差一个数量级。任何对外脚本若直接读 `experiments/`，拿到的不是论文数字。

## 复现论文表格（从 clone 到 Table 1）

```bash
git clone https://github.com/ml-universe/FOLoRA.git && cd FOLoRA
```

> **先读 [`docs/实验产物溯源.md`](docs/实验产物溯源.md)**：它说明哪些 run 是论文口径、
> 哪些是历史协议，以及当前正在进行的 O-LoRA / InfLoRA 重跑（`*_fix1` 标签）。
> 论文表格的数字全部来自 `reports/ncm/`，**不是** `experiments/*/results.json`。

```bash
# 1) 训练（幂等 + 断点续训；也可用 run_full_queue 走全量队列）
python -m scripts.run_single --benchmark cifar100 --num_tasks 20 \
       --method folora_v2 --seed 0 --folora_lambda 3 --folora_topk 64

# 2) 生成 NCM 缓存（零 GPU，读 checkpoint 冻结特征）→ reports/ncm/
python -m scripts.eval_ncm_sweep --benchmark cifar100
python -m scripts.eval_ncm_sweep --benchmark imagenetr

# 3) 显著性检验（默认 --source ncm + paired，即论文口径）→ reports/significance_*.md/.json
python -m scripts.significance --benchmark all

# 4) 生成论文表格 → paper/tables/（脚本内 OUT 常量；切勿改成 reports/）
python -m scripts.make_paper_tables
```

> 复现论文主配置**必须显式传参** `--folora_lambda 3 --folora_topk 64`：代码默认值仍是
> v2 全网格扫描期的旧值（300 / 0），与论文主配置不同（见下「方法」一节）。

## 显著性检验

`scripts/significance.py` 算 FOLoRA vs 各基线的检验，输出 `reports/significance_*.md` 与 `.json`。

- **默认数据源 = `--source ncm`**，即读 `reports/ncm/{benchmark}/*.json`（论文协议：冻结特征 +
  最近类均值）。该缓存由 `scripts/eval_ncm_sweep.py` 生成
- `--source results` 退回旧口径（线性头 CIL，读 `experiments/*/results.json`），仅供对照
- **默认检验口径 = paired t-test（配对 t 检验，`--test paired`）**，这才是论文正文的口径。
  各方法 seed 已由 `scripts/run_seed_align.py` 对齐
- `--test welch` 改用独立样本 Welch t 检验（`equal_var=False`）。它检验的是「两组独立样本」
  而非「配对差值」，**换口径必须在论文正文同步写明**
- `--check-paper` 把实测 p 值与论文正文引用的数值逐条对照（`*` 表示 p<0.05）
- `--all-configs` 扫全部 config（含消融与 EWC 调优），而非只跑主表

> **产物语义陷阱（曾致论文印错数字）**：`--all-configs` 输出的 `.json` 是
> 「**主配置 vs 各消融变体**」表，其 `ours_mean` 恒为主配置（加权 λ=3, k=64）的值，
> **不是**变体自己的 FOLoRA 臂。因此「等权 λ=10」那一行比的是
> 「加权 **λ=3** vs 等权 **λ=10**」，不能读成同 λ 下的加权/等权对照。
>
> 自 2026-10-03 起，每条记录都显式写入 `ours_tag` / `ours_method` / `ours_proto`
> （以及 `base_proto`），读产物时**先看 `ours_tag` 再读数值**：
> `ours_tag="v2f_l3_k64"` 与 `baseline_tag="v2f_eq_l10_k64"` 一眼可见非同 λ 对照。
> 旧产物（无 `ours_tag`）见桌面备份目录。若要真正做同 λ 对照，须回
> `reports/ncm/` 逐 seed 配对复算。

## 方法

| method | 说明 |
|---|---|
| `seq`   | 朴素串行 LoRA（无保护，下界） |
| `ewc`   | EWC-LoRA（对角 Fisher 惩罚 LoRA 参数） |
| `olora` | O-LoRA（每任务独立 adapter + 正交初始化） |
| `inflora` | InfLoRA（每任务子空间 + 与历史子空间正交的投影） |
| `l2p`   | L2P（prompt 池 + 余弦 top-k 选择 + key loss） |
| `coda`  | CODA-Prompt（prompt 组件 + 注意力软加权；未加其正交正则，属忠实简化） |
| `folora`  | FOLoRA **v1**（历史版本，正则项退化；**不要用它跑实验**，仅用于读旧 checkpoint） |
| `folora_v2`| **FOLoRA（本文方法：参数 Fisher 加权的软正交投影）** |

> **论文主配置**：`folora_v2` 的 `λ=3`、`k=64`（预注册主配置，见论文 tab:ablation）。
> 注意 `src/peft_cl/utils/config.py` 里的**代码默认值仍是 v2 全网格扫描期的旧值**
> （`folora_lambda=300`、`folora_topk=0`＝取满秩），**与论文主配置不同**。复现论文
> 必须显式传参：`--folora_lambda 3 --folora_topk 64`。

## 关键超参

- `--lora_rank` LoRA 秩（默认 16）
- `--folora_lambda` FOLoRA 正交正则强度 λ（**代码默认 300 = 旧扫描值**；论文主配置为 3）
- `--folora_topk` 保护方向数 top-k（0 = 取满 r 个方向；论文主配置为 64）
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
  fisher/    逐样本 Fisher 估计（对角 / 输出核 / 参数梯度低秩）
  methods/   seq / ewc / olora / l2p / coda / inflora / folora(v1) / folora_v2(本文方法)
  trainer/   持续学习训练循环（断点续训）
  metrics/   平均准确率 / 遗忘 / 前向迁移
  utils/     种子、路径、配置、原子写 JSON 等公共工具
scripts/     下载、单次运行、队列调度、NCM 评估、生成表格、显著性检验
paper/       LaTeX 论文（方法 + 理论推导 + 实验）；tables/*.tex 是生成产物，勿手改
docs/        实验产物溯源、方法总结、答辩讲稿
tests/       pytest 单元测试（CPU）
experiments/ 训练产物（checkpoint.pt / results.json）。体积大，未纳入版本库
reports/     NCM 逐 run 缓存 + 汇总 + 显著性报告 —— 论文表格的唯一数据源
```

> `methods/folora.py` 是 v1（历史负结果，正则近乎无效），论文方法在
> `methods/folora_v2.py`；两者同名不同物，见各自文件头。

## 测试

```bash
python -m pytest tests/ -q
```
