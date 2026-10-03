"""NCM 协议下的 λ 重新调优网格（断点续训友好、幂等）。

背景（为什么必须重调，而不是沿用原超参）
--------------------------------------
原先 FOLoRA 的 λ=300、k=16 是在**「随任务增长的可训练线性头」做 CIL** 的旧指标下
选的。那个指标下 CIL 准确率被头主导（8~11%），几乎看不到正则强度的影响，所以
选出的超参是任意的。换到领域标准的 **NCM（冻结/适配特征 + 最近类均值）** 协议后，
同一个 checkpoint 重评估得到的 λ 曲线（20/5/16 主协议，seed0）是：

    FOLoRA v2 (k=16)  λ=10 → 77.61 | λ=30 → 77.47 | λ=100 → 76.82
                      λ=300 → 73.61±1.48 (n=10) | λ=1000 → 74.19
    EWC-LoRA          λ=1  → 67.60 | λ=10 → 73.02 | λ=30 → 75.67±0.84 (n=5)
                      λ=1000 → 77.22±1.08 (n=10)

两者方向相反：FOLoRA 要**更小**的 λ（原默认 300 严重过正则），EWC 要**更大**的 λ。
向小的一侧延伸（λ=1、3）已由本网格完成，并找到了拐点——见下方 TAIL_CONFIGS 前的修正说明。

**聚合口径警告**：读 `reports/ncm/` 缓存做任何比较前，必须回读 `<run_dir>/config.json`
校验 `num_tasks==20 and epochs==5 and lora_rank==16`。只查 `num_tasks` 不够：
`v2_*` 那批是 **10 任务 / 2 epoch / rank 8**（完全不同、容易得多的协议，数值虚高约 4 点），
`*/pilot20` 那批是 `num_tasks=20` 但 `epochs=20`。踩过两次坑，别再踩。

为什么必须多 seed
----------------
实测**同配置同 seed** 的两次 run 也会差 1.3 个点（`ewc/default` 与 `ewc/ewc_lam100`
都是 λ=100、seed0，NCM 下 77.26 vs 75.97；bf16 AMP + cuDNN autotune 下不是逐位可复现）。
因此**单 seed 的 1 点级比较没有意义**——必须多 seed 取均值，本网格统一 5 seeds。

网格设计
--------
- FOLoRA v2：λ ∈ {1, 3, 10} × k=64 × seed 0..4（k=64 在 λ=10 上比 k=16 高 0.94）
- EWC-LoRA：λ ∈ {1000, 3000} × seed 0..4
按 **seed 轮次交错**执行（先全场 seed0、再全场 seed1……），这样即使中途断电/放弃，
已完成的也是一个「按 seed 平衡」的设计，而不是「前两个配置全跑完、后面的一个没有」。

用法（项目根目录，建议 detached + 日志）：
  python -u -m scripts.run_lambda_grid > reports/lambda_grid.log 2>&1
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# 基准不再是模块级常量：`transfer` 批跑在 imagenetr 上。用「按批查表」而不是可变全局，
# 是因为 supervise_all.py 会**在自己的进程里** import interleaved_runs/is_finished 来数
# 待跑数——若基准是全局量，那条路径会拿 cifar100 拼目录，把已完成的 INR run 判成「没跑」
# 而反复重跑（或反过来，把别的 run 判成已完成而漏跑）。默认值仍是 cifar100。
DEFAULT_BENCHMARK = "cifar100"
BENCHMARK_BY_BATCH = {"transfer": "imagenetr", "olora_orth_inr": "imagenetr",
                      "inflora_faithful_inr": "imagenetr",
                      # 2026-10-01 收尾批：三个跑在 ImageNet-R 上的批，**必须显式登记**。
                      # 漏登记的后果不是崩溃而是「静默漏跑」（见下方 is_finished 的 docstring）。
                      "transfer_ext": "imagenetr",
                      "coda_faithful_inr": "imagenetr",
                      "prompt20_topup_inr": "imagenetr",
                      # 2026-10-01 收尾H：EWC 的 INR λ 曲线补到 n=10，同样必须登记。
                      "ewc_inr_topup": "imagenetr"}
EPOCHS = 5
# 每任务的训练轮数也可按批覆盖：prompt 方法在论文里报的是 20 epoch（见 05_experiments
# setup「prompt-based methods are trained for 20 epochs per task」），所以补齐 prompt
# 基线的忠实配置时必须能改 epoch，而不是靠 `--epochs` 在命令行里出现两次赌 argparse 取
# 最后一个（能跑，但静默且脆）。
EPOCHS_BY_BATCH = {"coda_faithful": 20,
                   "inflora_faithful": 20, "inflora_faithful_inr": 20,
                   # 2026-10-01 收尾批：prompt 方法按论文 setup 训 20 epoch，
                   # 所以补 seed 也必须用 20（用 5 补出来的不是同一行）。
                   "coda_faithful_inr": 20,
                   "prompt20_topup": 20, "prompt20_topup_inr": 20}
RANK = 16
SEEDS = (0, 1, 2, 3, 4)


def benchmark_of(batch: str) -> str:
    """该 batch 跑在哪个基准上（未列出的批一律 cifar100）。"""
    return BENCHMARK_BY_BATCH.get(batch, DEFAULT_BENCHMARK)


def epochs_of(batch: str) -> int:
    """该 batch 的每任务轮数（未列出的批一律 EPOCHS=5）。"""
    return EPOCHS_BY_BATCH.get(batch, EPOCHS)

# (method, tag, 额外 CLI 参数, seeds) —— 顺序即优先级（seed 轮次内按此序）
# 待验证的候选配置排前面：万一时间不够，最关键的证据已经拿到。
CONFIGS = [
    ("folora_v2", "v2f_l1_k64",  ["--folora_lambda", "1", "--folora_topk", "64"], SEEDS),
    ("folora_v2", "v2f_l10_k64", ["--folora_lambda", "10", "--folora_topk", "64"], SEEDS),
    ("folora_v2", "v2f_l3_k64",  ["--folora_lambda", "3", "--folora_topk", "64"], SEEDS),
    ("ewc",       "ewc_lam1000", ["--ewc_lambda", "1000"], SEEDS),
    # EWC 的 λ 曲线在 1000 处仍在上升但已趋平，3000 只探 3 个 seed 确认是否见顶
    ("ewc",       "ewc_lam3000", ["--ewc_lambda", "3000"], (0, 1, 2)),
]

# ---------------------------------------------------------------- 第二批（低优先级）
# 「λ 鲁棒性」补充网格。**不在关键路径上**，排在别处都跑完之后（supervise_all 阶段 6）。
#
# 为什么要它：如果 λ 网格最终显示 FOLoRA 只是「追平」EWC，论文的卖点就要换成
# **对 λ 误设的鲁棒性**——这恰好是端侧持续学习真正在意的事（没法为每个数据集重调 λ）。
#
# **2026-09-23 修正**：原先这里写「FOLoRA 波动 3.4 点 vs EWC 10.4 点，差 3 倍」，
# 那是**在不匹配的 λ 区间**上比的（FOLoRA 取 [10,1000]，EWC 取 [1,1000]；
# EWC 的下限 67.60 来自 λ=1 seed0，而 FOLoRA 在 λ<10 处没有数据）。按口径校验后的
# 20/5/16 数据在**匹配区间**重算：
#
#     区间 [10,1000]: FOLoRA 极差 4.00 vs EWC 5.00   （FOLoRA 略好）
#     区间 [30,1000]: FOLoRA 极差 3.86 vs EWC 2.35   （**EWC 更好**）
#
# 即「3 倍鲁棒」不成立，λ≥30 区域 EWC 反而更稳（宽平台：λ=1000 n=10 = 77.22±1.08，
# λ=3000 seed0 = 77.61）。**不要把这个卖点写进论文**，除非主网格给出 FOLoRA
# λ∈{1,3,10} 的 5-seed 曲线后能用同一区间重新论证。
#
# 但**曲线两端的多种子证据不足**：FOLoRA 低端（λ=1/3/10）由第一批网格补齐 5 seeds，
# 高端 λ=100/1000 只有 seed0；EWC 低端 λ=1/10 只有 seed0，高端由第一批补齐。
# 「极差」由两端定义，所以这里补的正是缺的那几端，每档 3 seeds（效应量 ~7 点，
# 3 seeds 的 SE≈0.75 足够分辨，不必上 5）。
# ---------------------------------------------------------------- 等权消融（低优先级）
# 论文 Highlights 第 2 条写的是「importance-weighted orthogonalization outperforms
# equal-weight protection」——**这条目前没有任何实验支撑**（现有消融只有 λ 和 k）。
# 审稿人只要懂这个方法就会问「去掉 σ² 加权会怎样」，所以必须补。
#
# 对照的构造方式很关键：只在 `regularization_loss` 里把各方向权重齐次化成 σ̄，
# **方向与子空间完全不变、惩罚总尺度也保持不变**，所以两边可以在同一个 λ 下直接
# 比较，不需要为等权版重扫 λ。细节见 `FOLoRAv2Method._equalize_row_weights` 的 docstring。
#
# 三个 λ 各跑 3 seeds：主网格选出的最优 λ 必在 {1,3,10} 中，这样无论哪个 λ 赢，
# 都有**同 λ 的等权对照**可直接对照，不必等主网格结果再串行补跑。
UNWEIGHTED_CONFIGS = [
    ("folora_v2", "v2f_eq_l1_k64",
     ["--folora_lambda", "1", "--folora_topk", "64", "--folora_weighted", "0"], (0, 1, 2)),
    ("folora_v2", "v2f_eq_l3_k64",
     ["--folora_lambda", "3", "--folora_topk", "64", "--folora_weighted", "0"], (0, 1, 2)),
    ("folora_v2", "v2f_eq_l10_k64",
     ["--folora_lambda", "10", "--folora_topk", "64", "--folora_weighted", "0"], (0, 1, 2)),
    # **2026-09-25 用户拍板「补」新增**：等权臂 λ=30。
    # 理由——到 18:16 的配对数据显示，等权 vs 加权的差距随 λ **单调收窄**：
    # 同 seed0 上 +5.38(λ=1) → +4.14(λ=3) → +2.77(λ=10)。这引出**目前对
    # Highlights 第 2 条最硬的反驳**：「等权只是需要更大的 λ，收益来自正则强度
    # 而非分配方式」。而两臂精度在 [1,10] 内都**单调上升、无内部极值**，
    # 于是**等权臂的最优 λ 可能在 10 以上**——万一等权在 λ=30 冲到加权之上，
    # 第 2 条就塌了。现有数据**无法排除**这一点，所以必须补这个更远的点。
    # 反证据（若 λ=30 给出平台则成立）：等权 λ=10(75.78) ≈ 加权 λ=1(75.45)，
    # 即等权要 ~10 倍 λ 才够到加权在 λ=1 的水平。
    # 注意：这**不动主配置的 λ=3 冻结**（§13.3 规则 4 管的是主配置不因某些 seed
    # 好看而换 λ），只是给等权臂的 λ 曲线补一个更远的点。
    ("folora_v2", "v2f_eq_l30_k64",
     ["--folora_lambda", "30", "--folora_topk", "64", "--folora_weighted", "0"], (0, 1, 2)),
]

# **2026-09-25 扩充**（用户指令「力保数据真实最优」）。原表只给 EWC 的 λ=1/λ=10 各补
# seeds (1,2)，即 n=3；而 §11.5 要复活「λ 鲁棒性」这条卖点时留的条件是**同一区间**比较，
# 前提是**匹配的 n**——§11.0.1 实测「同 seed 同配置重跑就差 1.3–1.5 点」，所以 n=1 或 n=3
# 的极差无论多大都不构成证据。故把 EWC 的 λ∈{1,3,10} 全部扩到与 FOLoRA 主网格**同 seed
# 的 SEEDS=(0..4)**，这样两侧可以做**配对**检验（同 seed 相减），而不是只比均值±std。
#
# **λ=3 是新增的，原网格缺失**：原 λ 网格是 1,10,30,100,…（对数间隔，跳过了 3），
# 于是「FOLoRA λ=3 是同区间里最优点」这句话在 EWC 侧**没有对应点可比**——
# 论文主配置恰好就是 λ=3（§13.3 冻结），这个洞必须补上。
#
# 代价：+9 run ≈ 4.5h（λ=1 补 2、λ=10 补 2、λ=3 全新 5）。
# folora 的两个 k=16 点保持 (1,2)：它们是旧的 k=16 曲线，不参与 k=64 的匹配区间论证。
ROBUSTNESS_CONFIGS = [
    ("folora_v2", "v2f_l100_k16",  ["--folora_lambda", "100", "--folora_topk", "16"], (1, 2)),
    ("folora_v2", "v2f_l1000_k16", ["--folora_lambda", "1000", "--folora_topk", "16"], (1, 2)),
    ("ewc",       "ewc_lam1",      ["--ewc_lambda", "1"],  SEEDS),
    ("ewc",       "ewc_lam3",      ["--ewc_lambda", "3"],  SEEDS),
    ("ewc",       "ewc_lam10",     ["--ewc_lambda", "10"], SEEDS),
    ("ewc",       "ewc_lam300",    ["--ewc_lambda", "300"], (1, 2)),
]

# ---------------------------------------------------------------- λ=0 判决对照（高优先级）
# **修正（2026-09-23）**：本文件原先写的「λ 曲线到 λ=1 仍单调上升、拐点未现」是**错的**。
# 那个结论来自 `v2_l1_k64`=79.65 / `v2_l1_k16`=79.48，而这两个 run 的 config 是
# **num_tasks=10, epochs=2, lora_rank=8**——一个完全不同、容易得多的协议，
# 根本不能和 20 任务主协议放在一起比。凡是读 NCM 缓存做聚合的地方，都必须回读
# `<run_dir>/config.json` 校验 20/5/16（只查 num_tasks 不够，pilot20 那批 nt=20 但 ep=20）。
#
# 主协议下（20/5/16）的真实 λ 曲线是有拐点的：
#
#     k=16 (seed0):  λ=10→77.61  30→77.47  100→76.82  300→73.61±1.48(n=10)  1000→74.19
#     k=64 (seed0):  λ=1 →75.45  10→78.55
#
# 即峰值在 λ≈3~10，**λ=1 已经开始下降**（75.45 明显低于 λ=10 的 78.55），
# 所以「λ→0 会不会更好」这个担心基本可以排除，主网格的 λ∈{1,3,10} 已经圈住拐点。
# 因此 λ=0.1 / λ=0.3 两个探针**取消**（原计划 10 个 run ≈ 8h，纯属追一个不存在的现象）。
#
# 保留 λ=0 一个档（5 seeds），因为它是**消融表的锚行**，而且顺带排除一个真实的方法论隐患：
# `after_task` 里为估 Fisher 会多跑若干次 forward/backward，若那些 pass 动到了 dropout/LayerNorm
# 的 RNG 状态，λ=0 就不会精确等于 Seq-LoRA，整个「加正则 vs 不加」的对比就被混淆了。
# 实测 λ=0 若落在 Seq-LoRA（现成 n=10 = 69.31）的噪声带内，这个隐患就被排除；
# 若显著偏离，说明代码路径本身有副作用——那比 λ 选多少重要得多。
TAIL_CONFIGS = [
    ("folora_v2", "v2f_l0_k64", ["--folora_lambda", "0", "--folora_topk", "64"], SEEDS),
]

# ---------------------------------------------------------------- k 消融（中优先级）
# k 是唯一**没有多 seed 证据**的超参数：现有 k=16 与 k=64 的对比全是单 seed，
# 而实测同 seed 同配置重跑就差 1.3~1.5 点，所以此前所有 k 的排序都没有意义。
# 审稿人看到消融表只有 λ 一行一定会问 k。
#
# 固定 λ=10：这是主协议下 k=64 的最好证据（78.55），也落在 k=16 的峰值区（λ=10~30），
# 两端都合理。**原定 λ=1 已按上面的修正改掉**——λ=1 在 k=64 上只有 75.45，
# 明显在下降侧，拿它做消融会把 k 曲线整体压到次优区间。
# 若主网格 5 seeds 最终选出 λ=3，再补 k=16/128 在 λ=3 的 10 个 run 即可。
# k=64 的 5 seeds 由主网格免费给出，所以这里只补两端，即得 {16,64,128} 三点曲线。
#
# 这条曲线还有一个副产品：**k 是 FOLoRA 显存的唯一驱动因素**。重要性核在每层是
# (k, d) 的矩阵，d≈61440，实测 k=16 → 47.8 MB、k=64 → 182.8 MB 的 checkpoint 增量，
# 而 EWC 的整条对角 Fisher 只要 ~2.9 MB。也就是说 FOLoRA 花 k 倍于 EWC 的训练期状态
# 换取「捕捉方向相关性」的能力——这是个**必须如实披露的取舍**，而 k 曲线正是它的
# Pareto 前沿。论文只能主张可训练参数固定/推理零开销，**绝不能主张比 EWC 省内存**。
# k=16 若追平 k=64，就是「更省内存且不掉点」的严格占优配置，故 k=16 排在前面。
KDIM_CONFIGS = [
    ("folora_v2", "v2f_l10_k16",  ["--folora_lambda", "10", "--folora_topk", "16"],  SEEDS),
    ("folora_v2", "v2f_l10_k128", ["--folora_lambda", "10", "--folora_topk", "128"], SEEDS),
]

# ---------------------------------------------------------------- O-LoRA 聚合公平性对照
# `MultiLoRALinear.forward` 把 T 个 adapter 直接相加，**这正是 O-LoRA 论文原式**
# （W0 + Σ_i B_i A_i），不是复现 bug。但在 20 任务 × rank 16 下 ΔW 的秩上限是
# 320/768 ≈ 42%，NCM 特征被推歪，实测只有 64.79±2.08——低于 Seq-LoRA(69.31) 和
# SimpleCIL(70.31) 两条地板线，审稿人会直接质疑「你的 O-LoRA 复现是坏的」。
#
# 这里跑 mean 聚合（ΔW = (1/T) Σ B_i A_i）作为对照，用来回答
# 「是不是聚合方式的问题」。两个走向下都需要：
#   - mean 明显更好 → 主表报 mean，并注明原式数值（附录）
#   - mean 同样差（很可能落到 SimpleCIL 附近，因为 1/T 缩放会让适配器几乎不起作用）
#     → 说明「任何聚合方式都救不回来」，这本身是一句**有数据支撑**的结论，
#       比只报一个孤零零的低分强得多
# 3 seeds 够：效应量若存在是 5 点级，SE≈1.1 足以分辨。
OLORA_AGG_CONFIGS = [
    ("olora", "olora_mean", ["--olora_aggregate", "mean"], (0, 1, 2)),
]

# ---------------------------------------------------------------- O-LoRA 正交性约束（忠实版）
# **为什么必须有这一批**：现在主表里的 "O-LoRA" 只做了**正交初始化**，训练期没有正交性
# 惩罚（`OLoRAMethod.regularization_loss` 原来恒返回 0）。而 O-LoRA 原文的目标是
#     Σ log p(y|x) + λ₁ Σ_{i<t} ‖A_tᵀ A_i‖_F² ，
# 即正交性是**训练期约束**，不是初始化。两个后果：
#   1. 只初始化、不加约束，第一个梯度步 A_t 就离开正交互补空间，"正交子空间"这个前提
#      实际不成立 —— 我们测的是比原方法**更弱**的东西。基线被做弱是审稿人最容易抓的点，
#      和 EWC 把 λ 调小属于同一类错误（那一处已在 §3/§5 修过）。
#   2. 若把这个更弱的变体当成"O-LoRA 在 CIL 下不行"的证据，等于用一个未忠实复现的
#      基线去支撑我们的动机段 —— 这是最不该留给审稿人的把柄。
#
# **λ₁ 取值**：既然初始化已使惩罚为 0，这一项的作用是"保持在正交子空间附近"的势垒，
# 而非从零把 A 拉正交，所以在很宽的 λ₁ 范围内都应当有效。取 {0.1, 1.0} 两侧夹逼
# （文献建议起手 0.1，先验掉得多则往 0.5~1.0 加），便于说明"不是只试了一个值"。
# 3 seeds/配置 = 6 run ≈ 4.8h。这是**基线行**的忠实性成本，不是我们的方法在做调参。
#
# **可砍**：若时间不足，退路是**如实降级命名**——把主表那一行写成
# "O-LoRA-style orthogonal init (no training-time constraint)"，并在正文说明它与原文
# 的差别，而不是继续称其为 O-LoRA。绝不能默默把弱变体当 O-LoRA 报。
OLORA_ORTH_CONFIGS = [
    ("olora", "olora_orth_l0.1", ["--olora_orth_lambda", "0.1"], (0, 1, 2)),
    ("olora", "olora_orth_l1",   ["--olora_orth_lambda", "1.0"], (0, 1, 2)),
]

# ---------------------------------------------- O-LoRA 正交性约束（**INR 版**）
# **2026-09-27 用户拍板「补」新增**（起因：CIFAR 版 3j 跑完后发现两表不自洽）。
#
# 补的是**两张主表里「O-LoRA」指的不是同一个东西**：`OLORA_ORTH_CONFIGS` 只在
# `cifar100` 上跑（该批 benchmark 已核），于是 CIFAR 表有 4 个 O-LoRA 变体，
# 而 INR 表里 O-LoRA 仍只有 `default`（**未实现训练期正交项**、原文的缺失变体）。
# 这正是 5b 当初要修的那类「两表自相矛盾」的镜像——审稿人只要并排看两张表就会发现。
# INR 现状：`olora/default` 44.79 (n=5) vs seq 地板 56.58 (n=5)，
# 配对 Δ=−11.79, p=0.00047, 逐 seed 全负——比 CIFAR 的 −4.51 更难看且显著，
# 更经不起「你把基线做弱了」这一问。
#
# **λ 与 seed 与 CIFAR 版逐项一致**（λ∈{0.1,1.0}，seeds (0,1,2)）：若两表用不同的
# λ 或不同的 seed 集，那"同一变体"仍然不是同一个配置，等于把矛盾从"变体不同"
# 挪到"参数不同"，没有修好。
#
# 性质必须是"设计"而不是"事后补测"：本批 6 个 run 加入时**全部 pending**，
# 与 CIFAR 版的结果无任何依赖（不因为它在我们有利/不利而增删）。
# **这 6 个 run 的 λ 值与 seed 集不得因结果好坏而增删。**
#
# 报告规则：INR 表的 O-LoRA 行与 CIFAR 表一样**四个变体全报**；不得因为
# 忠实版更高或更低而只留一行。
OLORA_ORTH_INR_CONFIGS = [
    ("olora", "olora_orth_l0.1", ["--olora_orth_lambda", "0.1"], (0, 1, 2)),
    ("olora", "olora_orth_l1",   ["--olora_orth_lambda", "1.0"], (0, 1, 2)),
]

# ---------------------------------------------------------------- T=50 规模轴（决定性实验）
# **这是唯一能把「+0.5 点」变成结构性结论的实验。**
#
# 动机：FOLoRA 的正则在**跨任务累积**一个受保护子空间——任务越多，新任务需要避开的
# 方向越多，**方向相关性**就越关键，而对角 Fisher（EWC）在这种情形下会饱和。
# 若 FOLoRA 相对 EWC 的差距随 T 增长（T=20 时 +0.5，T=50 时 +2），论文就有了
# 结构性声明：「优势随任务数放大」，而不是靠单点运气。
#
# 为什么 T=50：CIFAR-100 有 100 类，`make_class_order` 要求 num_classes % num_tasks == 0，
# 100/50 = 2 类/任务成立。而且 T=50 正是 L2P / CODA-Prompt 的 **CIFAR-100 B/50 标准设定**，
# 有现成的文献锚点，不是我们自创的protocol。
#
# **两边都扫 λ**：若只给 FOLoRA 调 λ 而 EWC 用 T=20 的旧值，会被质疑「不公平」。
# FOLoRA 取 {3,10,30}（覆盖 T=20 的拐点区），EWC 取 {100,1000}（覆盖它的平台区），
# 各自在自己的合理区间内取最好，比较才是公平的。
# **2026-09-26 22:4x 用户拍板「全补」，新增 EWC λ=300（见下方注释）后为 7 配置 × 5 seeds
# = 35 run**，合计约 35~42h。
# 总训练样本量与 T=20 相同（都是 100 类 × 5 epoch），只是多了 30 次 Fisher 估计，
# 故单 run 约 1.0~1.2h。
SCALING_CONFIGS = [
    ("folora_v2", "t50_l10_k64",   ["--folora_lambda", "10", "--folora_topk", "64"], SEEDS),
    ("ewc",       "t50_ewc_l1000", ["--ewc_lambda", "1000"], SEEDS),
    ("folora_v2", "t50_l3_k64",    ["--folora_lambda", "3",  "--folora_topk", "64"], SEEDS),
    ("ewc",       "t50_ewc_l100",  ["--ewc_lambda", "100"], SEEDS),
    ("folora_v2", "t50_l30_k64",   ["--folora_lambda", "30", "--folora_topk", "64"], SEEDS),
    # **2026-09-26 22:45 用户拍板「全补」新增**：EWC λ=300。
    #
    # 补的是**我们自己制造的、可被当场证伪的洞**：§24.2 的敏感性表（必须进论文）
    # 说 EWC 在 T=20 的最优点是 λ=300（配对 77.98 > λ=1000 的 77.32 > λ=100 的 77.68
    # 中的 77.98），而 T=50 的 EWC 网格原本是 {100, 1000} —— **恰好从它自己的最优点
    # 两侧擦过去**。审稿人不需要复现任何实验，对着两张表就能指出这一点。
    #
    # **设计对称性**（加它的主要理由，而非"多跑一个点"）：补后
    #     FOLoRA {3, 10, 30}  —— 3 点，括住其 T=20 最优点 λ=10
    #     EWC    {100, 300, 1000} —— 3 点，括住其 T=20 最优点 λ=300
    # 两个方法的 λ 网格**点数相同、且各自括住各自的最优点**，结构完全平行。
    #
    # **性质必须是"设计"而不是"事后补测"**：本行加入时 stage 6 的 30 个 run
    # **全部 pending、一个都没启动**（预计 09-28 凌晨才轮到），故它属于预先注册的
    # 网格。**若在 T=50 出结果之后再补，就是事后选点** —— 因此本行一旦加入，
    # 其 5 个 run 的 λ 值不得因结果好坏而增删。
    #
    # 风险已知并接受：若 λ=300 在 T=50 上成为 EWC 最优且不弱于 FOLoRA 最优，
    # 我们的 T=50 结论会从"显著"掉回"平价"。**但它不可能让我们失去任何现有结论**
    # —— T=20 的平价、λ 稳健性、k=64 稳定性、v1→v2 机制发现，全都不依赖这个数；
    # 它唯一能杀掉的是我们**还没有**的那个"优势随 T 放大"的期望。
    ("ewc",       "t50_ewc_l300",  ["--ewc_lambda", "300"], SEEDS),
    ("seq",       "t50_seq",       [], SEEDS),
]

# ---------------------------------------------------------------- 界的实证探针（1 run）
# 04_theory.tex 有两处承诺要在 Sec. discussion 里做实证，而 sec:discussion 里**一条都没有**
# （已 grep 确认）：(1) 高阶项相对二阶项可忽略；(2) Assumption 1 的违背量级。承诺了不做
# 比不说更伤。这个 run 把两个数真的测出来，让承诺变成事实。
#
# 为什么必须专门跑一个 run：`after_task` 会把 ref_params 覆盖成当前任务快照，所以训练
# 结束后落盘的 checkpoint 里 ref_params == model_state（已实测 ‖cur−ref‖ = 0），事后从
# 单个 checkpoint 拿不到位移。可用的三元组 (θ_{t-1}, θ_t, F̄_{<t}) 只在 after_task **之前**
# 同时存在于内存，所以探针必须挂在训练过程里（见 utils/bound_probe.py）。
#
# λ=10 / k=16：**这不是论文主配置**（主配置是预注册的 λ=3 / k=64）。选它是为了让
# 影响半径与内存都更小、单 seed 也能看清曲线形状：k=16 档在 NCM 协议下于 λ∈{3,10}
# 达峰（见 paper §5.3），λ=10 就落在该档峰上。论文 §4 报的 bound 各项数值**取自这个
# 点**（原文写明 "FOLoRA at λ=10, k=16, seed 0 on Split CIFAR-100"），所以改动本探针
# 的配置就必须同步 §4 的数字，不能只改这里。（2026-10-03 更正：旧注释称其"与 NCM 协议
# 下选出的默认配置一致"，那是在主配置还是 λ=10/k=16 时写的，现已不成立。）
# 只 1 个 seed：这是机制验证（曲线形状），不是性能比较，不需要多 seed。
# 代价约 1h 训练 + 每任务边界几分钟（探针只做前向，无反向）。
BOUND_CONFIGS = [
    ("folora_v2", "probe_l10_k16",
     ["--folora_lambda", "10", "--folora_topk", "16", "--bound_probe"], (0,)),
]

# ------------------------------------------------- ImageNet-R 的「冻结点迁移」网格（10 run）
# 补的是一个**内部矛盾**，不是漏跑：主表有 INR 列，而 INR 上现存的 FOLoRA 只有
# `v2f_l300_k16`（λ=300, k=16，5 seeds）—— 正是 NCM 网格判定为「严重过正则」的那个点
# （CIFAR 上 73.61，比 EWC λ=100 低 3.6 点）。而论文 §ablation 又自己论证了 λ=300 次优。
# 于是一篇论文同时说「λ=300 不好」和「我们在 INR 上主报 λ=300」，这是审稿人对照两张表
# 就能看出的矛盾；它同时是 EWC 那处「基线被调小」缺陷的**镜像**（在 INR 上我们的方法
# 用了次优超参、基线用了别处的超参）。INR 的 run 只有 CIFAR 的一半代价（每任务 1200 vs
# 2500 张），5.5h 换掉这个矛盾，很划算。
#
# 协议口径：超参**在 CIFAR-100 上选、原样迁移到 INR**（对所有方法一致），所以每个方法
# 只跑一个点，不跑 INR 自己的 λ 曲线 —— 若在 INR 上也扫 λ，就变成「两个基准分别调参」，
# 与论文写的协议不符。冻结点来自 §13.3 规则 2（k=64, λ=3，判决前已定，不是看结果选的）
# 与 EWC 的 NCM 最优点 λ=1000。
#
# 为什么 EWC 也必须在列：INR 现存 EWC 是 λ=100（默认值），而 CIFAR 上 EWC 的 NCM 最优是
# λ=1000（77.22）。只迁 FOLoRA 不改 EWC，就是把「不公平」的方向反过来；两个都迁才对称。
TRANSFER_CONFIGS = [
    ("folora_v2", "v2f_l3_k64",   ["--folora_lambda", "3", "--folora_topk", "64"], SEEDS),
    ("ewc",       "ewc_lam1000",  ["--ewc_lambda", "1000"], SEEDS),
    # **2026-09-26 22:45 用户拍板「全补」新增**：EWC λ=300。
    #
    # 补的是**协议违规**，不是"多探一个点"。上面这段注释（写于 λ=300 的数据到手之前）
    # 把 EWC 的迁移点定为"NCM 最优点 λ=1000"；但 §24.2 的敏感性表后来测出 CIFAR 上
    # **EWC 的真正最优是 λ=300（配对 77.98 > λ=1000 的 77.32）**。于是按本段自己写的
    # 规则（「冻结点来自 CIFAR 选出的最优点，原样迁移」），INR 该迁的是 300，而不是 1000。
    #
    # 关键区别：**这不是"在 INR 上扫 λ"**（上面明确禁止了那样做，理由是会变成
    # "两个基准分别调参"）。它是**修正"迁哪个点"这一个决策**，仍然每个方法一个点。
    #
    # 报告规则（必须遵守，否则会造出新的内部矛盾）：INR 的 EWC **两行都报**——
    #   `ewc_lam300`  = 协议行（CIFAR 最优点，与 FOLoRA 的 v2f_l3_k64 同为迁移点）
    #   `ewc_lam1000` = 稳健性对照（原迁移点，已在表内，不得删）
    # **不得**因为 300 与 1000 谁高谁低而只报其中一行。注意 INR 上 EWC 从 100→1000
    # 是**上升**的（63.59→66.02），所以 300 大概率弱于 1000；若真如此，我们的 INR
    # 差距会变大，**这是"照实报"，不是"选了个弱基线"** —— 判据是它符合论文写定的
    # 迁移规则，而不是它对我们有利。
    ("ewc",       "ewc_lam300",   ["--ewc_lambda", "300"], SEEDS),
]

# ------------------------------------------- prompt 基线的「忠实配置」（3 run，最低优先级）
# 补的是**基线配置的混淆**：CODA-Prompt 在仓库里有两个 tag，而它们**同时**差两个变量——
#
#     coda/default   pool=100, len=8, 5 epoch   ← 原论文的几何（大组件池 + 注意力组合）
#     coda/pilot20   pool=20,  len=5, 20 epoch  ← L2P 的几何
#
# 论文 setup 写的是「prompt-based methods are trained for 20 epochs」，即主表取 pilot20
# —— 于是报出去的「CODA-Prompt」用的是 **L2P 的 prompt 几何**（config.py 自己也写着
# pool_size 就是「CODA 组件数」）。这与 O-LoRA「只做正交初始化却叫 O-LoRA」、EWC「λ 调小
# 到次优」是同一类问题：把一个更弱的变体当成该方法的成绩报出去。而且因为两个变量同时变，
# 连「20 epoch 是否更好」这个比较本身都不干净（L2P 那边的两个 tag 只差 epoch，是干净的；
# CODA 这边不是）。
#
# 为什么补的是 (pool=100, len=8, 20 epoch) 这一格：论文要报的是「该方法的配置」，即忠实
# 几何 + 论文自己声明的 epoch 数，两者同时成立的那一格，目前**不存在**。
# 实测代价（CIFAR，seed0 的 run.log 时间戳）：5ep×pool100 = 43 min；20ep×pool20 = 111 min。
# 所以 20ep×pool100 ≈ 4×43 ≈ **2.9 h/run**（pool=100 的注意力开销让每 epoch 比 pool20 贵），
# 3 seeds ≈ 8.7 h。这就是它排在**最后**的原因：它是唯一一个「不补也不影响结论、只影响
# 基线公平性叙事」的批，时间不够直接从尾部砍掉，一点不心疼。
# 只跑 CIFAR：按「在 CIFAR-100 上选配置、原样迁移」的统一协议，先在 CIFAR 上比出忠实几何
# 与 pilot20 孰强；只有当忠实几何胜出时，才需要再补 INR 的 3 个 run。
CODA_FAITHFUL_CONFIGS = [
    ("coda", "pool100_len8_ep20",
     ["--prompt_pool_size", "100", "--prompt_length", "8"], (0, 1, 2)),
]

# ================================ InfLoRA「训练预算对照」（用户 2026-09-28 拍板「补」）
# 补的是什么：**我们的 InfLoRA 复现在两个基准上都不可信**——
#   CIFAR：`inflora/default` n=5 = **70.14±0.26**，对**同 seed 地板**（seq seeds 0-4 = 69.33）
#          只有 **+0.81、t=+1.055、p=0.351**，5 个 seed 里 2 个为负 ⇒ 与地板统计不可区分；
#   INR  ：seed0 = **50.03**（`reports/ncm/imagenetr/inflora__default__seed0.json`），
#          **低于 INR 地板 56.58 达 −6.55** —— 一个 ICLR'24 的方法在我们表里输给朴素
#          序列微调，这是审稿人第一眼就会指着问「你们是不是复现坏了」的位置。
# 配置已核（`experiments/cifar100/inflora/default/seed0/config.json`）：epochs=5, lr=1e-3,
# lora_rank=16, batch_size=32 —— 与 seq/ewc/olora **完全同协议**，没有单独动过 InfLoRA。
# ⇒ 问题是「**统一协议 vs 方法非忠实**」这个经典两难，不是配置写错。
#
# ------------------------------------------------------------------ 这批做什么、不做什么
# **只改训练预算：epochs 5 → 20，其余（lr / batch / rank / alpha / seed / 任务划分）一字不改。**
# 理由：① 单变量，差异可归因；② 20 epoch 是本仓库**已存在的最长训练预算**（CODA 忠实配置
# 就是 20ep，见 `EPOCHS_BY_BATCH`），所以口径统一成「给基线我们所能给的最长预算」，而不是
# 为 InfLoRA 单独发明一套超参；③ 我没有 InfLoRA 官方 recipe 的凭据（论文不在手边），
# **因此绝不声称这批是「官方复现」**——它在论文里写成 *training-budget sensitivity control*。
#
# ------------------------------------------------------------------ 预注册（红线，同 §26.21/§26.25.3 规格）
# **本批 6 个 run 的 epochs(=20)、lr(不动)、seed 集(0,1,2) 与 tag 不得因结果好坏增删。**
# 加入时 6 个 run 全部 pending（已在 §26.27 记录 `--status` 输出），与任何已出结果无依赖
# ⇒ 性质是**事先设计**而非事后补测。
# **报告规则**：InfLoRA 行**两套都要报**（5ep 统一协议 n=5 / 20ep 对照 n=3），不得因为
# 20ep 更高或更低而只留一套；CIFAR 与 INR 两表同理。
# **诚实预警（写在这里，结果出来前）**：这批**可能把 InfLoRA 抬到接近甚至超过 FOLoRA**
# （INR 上 FOLoRA 66.92）。若如此，必须改叙事（收窄主张），**不得**因为不好看而不报。
#
# ------------------------------------------------------------------ 代价（实测重估，不是拍脑袋）
# 实测 5-epoch InfLoRA：CIFAR **1.12–1.14 h/run**、INR **0.76 h/run**（n≥2/3）。
# 逐任务拆开后训练只占约一半（CIFAR seed4：训练 ≈35 min、逐任务评估 ≈34 min，评估量随
# 已见任务数线性增长 6s→220s，**不随 epoch 数变化**）⇒ 20ep ≈ 训练×4 + 评估不变：
#   CIFAR ≈ **2.9 h/run**（3 run ≈ 8.7h）、INR ≈ **1.8 h/run**（3 run ≈ 5.3h）⇒ 合计 **~14h**。
# （09-28 我口头给的 8–10h 是误估——4× 训练量的直觉忽略了评估占比。以本注释为准。）
#
# 两个基准的 tag **故意取不同名**（`ep20` / `ep20_inr`），与 `olora_orth_*` 用同名不同：
# 同名会让 `is_finished(method, tag, seed)` 在漏传 benchmark 时**静默查错目录**（正是 §26.25.1
# 第 5 条记录的坑）。不同名是纵深防御，代价只是表里多一个后缀。
INFLORA_FAITHFUL_CONFIGS = [
    ("inflora", "ep20", [], (0, 1, 2)),
]
INFLORA_FAITHFUL_INR_CONFIGS = [
    ("inflora", "ep20_inr", [], (0, 1, 2)),
]

# ---------------------------------------------- S-12：主表两行补 seed 5-9（10 run，~5h）
# 补的是**论文陈述与实验事实不符**：论文 setup 写「CIFAR-100 上 10 seeds」（论文里这是
# 一个卖点——CL 的方差大，多 seed 是审稿人看重的严谨性），但 §13.3 规则 2 冻结出来的新
# 主配置（v2f_l3_k64）**只有 5 seeds**，因为它是 λ 网格跑出来的、网格只跑 SEEDS=0-4。
# EWC λ=1000 同理（它是网格里 EWC 侧的点，也只有 5 seeds）。于是论文声明 10 seeds 而
# 主表的这两行是 5 seeds —— 这是**陈述与数据不符**，比「方法弱」危险得多：审稿人只要
# 数一下 results.json 的目录就会要求解释，而任何解释都是承认陈述不实。
#
# 注意这**买不到显著性**：λ=3 与 EWC λ=1000 的效应量约 +0.46 点，加到 10 seeds 更可能
# 把判决钉在「平价」（§13.3 规则 4 允许平价，因为互补轴才是主贡献）。它的价值是让
# 论文的「10 seeds」这句话变成真的、并把配对检验的功效翻倍（n=5→10）。
#
# 只补这两行、不补别处的理由：主表的**其余**方法（L2P/DualPrompt/CODA/seq/LoRA…）是
# 早期跑的，本来就是 10 seeds——只有 λ 重调后新增的这两行是 5。补完主表就整体是 10。
# 与 `transfer` 批（INR 5 seeds）不冲突：INR 论文声明就是 5 seeds。
#
# 代价：CIFAR 每 run 约 25-30 min → 10 run ≈ 5h。
MAIN_EXT_CONFIGS = [
    ("folora_v2", "v2f_l3_k64",  ["--folora_lambda", "3", "--folora_topk", "64"], (5, 6, 7, 8, 9)),
    ("ewc",       "ewc_lam1000", ["--ewc_lambda", "1000"], (5, 6, 7, 8, 9)),
]

# ------------------------------ EWC λ=300 补齐到 5 seeds（5 run，~2.5h，用户 2026-09-24 决定）
# 补的是**最后一个还没关掉的「基线可能被选弱」口子**，性质和已修过的三处（EWC λ 调小、
# O-LoRA 弱变体、CODA 换几何）完全一样，只不过方向相反：这次是**基线的某个点可能更强，
# 而我们没测**。
#
# 事实：NCM 口径下 EWC 各点的均值是 λ=100 → 77.22 (n=10)、λ=300 → **78.02 (n=1)**、
# λ=1000 → 77.32 (n=3)、λ=3000 → 77.27 (n=3)。其中 **λ=300 是全仓库最高的 EWC 数字，
# 却只有一颗种子**（2026-09-04 跑的，已移到 reports/legacy_runs/ 保留）。在 seed0 上它
# 78.02 > 我们主配置的 77.83 —— 也就是说**在我们自己唯一能直接比的那颗种子上，我们是输的**。
# 我们冻结时把 λ=1000 当作 EWC 的比较点，如果 λ=300 补到 5 seeds 后确实更高，那么
# 「EWC 的最佳点是 λ=1000」这句话就是假的，而审稿人只要拿到代码搜一遍就能发现。
#
# ------------------------------------------------------------------ 冻结的选点规则
# **写在这里、在数据出来之前**（同 §13.3 的精神：防 garden of forking paths）：
# 1. EWC 的比较点 = **n≥5 的 EWC 点中 NCM 均值最高者**（λ=100 已 n=10；补完 λ=300 后
#    λ=300 与 λ=1000 都是 n=5）；
# 2. 若最高点不是 λ=1000、但与它的 5-seed 均值差 **< 0.20 点**（≈ 该基准 seed 间标准差的
#    量级以下），则主表**仍报 λ=1000**，附录给出完整 λ 曲线 —— 不因为 0.1 点的差就改主表
#    基线，那正是"事后挑一个好看的"；
# 3. 若差 **≥ 0.20 点**，主表**改报那个点**，配对检验一律对它做 ——
#    **即使这样会让我们的净胜变小、甚至翻成负**；
# 4. 绝不为了找更好的 EWC 点再去试新档位（λ=300 是最后一个）。
#
# 附带收获：新的 seed0 与已移到 `reports/legacy_runs/ewc_lam300_seed0_2026-09-04/` 的旧
# seed0 是**同配置、同 seed、跨代码版本**的一对，两者之差是「可复现性下限」的第二个独立
# 测量（第一个是 ewc/default 与 ewc_lam100 在 seed0 上的 77.26 vs 75.97 = 1.29 点）。
#
# 注意：跑了新 seed0 就必须删掉旧的 NCM 缓存 `reports/ncm/cifar100/ewc__ewc_lam300__seed0.json`
# ——缓存**只按路径判断是否已算过、不看 checkpoint 是否变过**（eval_ncm_sweep.py:95-98），
# 不删的话新 run 会静默沿用旧结果。已随旧目录一起移到 reports/legacy_runs/ 下。
EWC300_CONFIGS = [
    ("ewc", "ewc_lam300", ["--ewc_lambda", "300"], (0, 1, 2, 3, 4)),
]

# ============================================================================
# 2026-10-01 3n：EWC 的 λ 曲线补到同 n=10（12 run，~9.1h）
# 用户 2026-10-01 拍板：「遵守规则 + 补 3n 再报」。
# 存在理由（**数据之前**已由日志 §24.2 写下，本批只是执行，不是结果后的新主意）：
#   预注册规则 1 用「n≥5 的 EWC 点中均值最高者」选主表行，但 λ=300 只有 n=5、
#   λ=1000 有 n=10 —— 不同 n 的均值直接比是规则自身的漏洞（§26.41.2 已记）。
#   §24.2 数据前就写过修法：「**修法不是选一个好看的 p，而是把两边都补到同 n**
#   （3l 让两边都到 10，**3n 让 EWC 的 λ 曲线在同 n 口径下可比**）」。
#   3l 已执行完毕；3n 迄今未执行。本批补上。
# **为什么 λ=300 与 λ=3000 一起补、而不是只补赢我们的那一个**：
#   只补 λ=300 会被读成「挑 n」（专挑赢了我们的基线去补）。把整条 λ 曲线统一补到
#   n=10（λ=100 与 λ=1000 本来就是 10）才是「统一口径」这一件事本身。补完 n≥5 集合
#   = {λ=100(10), λ=300(10), λ=1000(10)}；λ=3000 到 10 仅用于附录曲线同口径可比。
# **方向不保证、也不允许回删**：λ=300 补到 10 后若回落 <λ=1000 ⇒ 规则 2 生效（主表
#   仍报 λ=1000）；若维持 >λ=1000 ⇒ 规则 3 生效（主表报 λ=300）。**两种结果都如实报**。
# ============================================================================
EWC_CURVE_TOPUP_CONFIGS = [
    ("ewc", "ewc_lam300",  ["--ewc_lambda", "300"],  (5, 6, 7, 8, 9)),
    ("ewc", "ewc_lam3000", ["--ewc_lambda", "3000"], (3, 4, 5, 6, 7, 8, 9)),
]

# ============================================================================
# 2026-10-01 收尾H：EWC 的 **ImageNet-R** λ 曲线补到同 n=10（15 run，~8.3h）
# 用户 2026-10-01 拍板：「全补」。
#
# 存在理由（看数据之前就成立，不是结果后的新主意）：
#   14a（`transfer_ext`）已把 INR 的 FOLoRA 主行补到 n=10，但 INR 的 EWC 仍全是 n=5。
#   后果有二，且都跟结果好坏无关：
#     (a) **配对检验用不上新增的 5 个 seed**（§13.3 规则 3 只在共同 seed 上配对），
#         14a 花的 3.2h 在 INR 上**没买到任何检验功效**——表里的均值变了，功效没变。
#     (b) INR 表变成「我方 n=10 / 基线 n=5」的不对称，正是 §26.41.2 自己指出过的
#         「不同 n 的均值直接比是规则自身的漏洞」。
#   CIFAR 侧已由 3n（`ewc_curve_topup`）补到三点同为 n=10；本批是**同一条逻辑的 INR 侧**。
#
# **三点全补、不挑**：只补 λ=1000（赢了我们的那个）会被读成「挑 n」。
#   补完 n≥5 集合 = {λ=100(10), λ=300(10), λ=1000(10)}，与 CIFAR 侧口径一致。
#   λ=100 现均值 63.59，落后 2.4 点以上，永远不可能赢得规则 1 的选点——
#   补它纯粹是为了「整条曲线同 n」，属于口径成本，不是赌注。
#
# **方向不保证，两种结果都如实报（不得回删任何已有 seed）**。已由 §26.48 用预测分布
#   模拟（20 万次）预先量化，写在这里防止事后重新解释：
#     * 对 λ=1000（主表行）的配对 Δ 预期从 +0.90 落到 **+0.35**，p 从 0.070 到 **≈0.29**，
#       **只有 3.1% 的概率反而变显著**（96.6% 变成明确平价）；
#     * 对 λ=300 预期从 p=0.0091 落到 **p≈0.15**（81% 概率失去显著）；
#     * 规则 3（λ=300 反超 λ=1000 且差 ≥0.20 ⇒ 主表行换人）约 **11%**。
#   这不是"让 p 变好看"的动作；它买的是**口径自洽**与**「seed 噪声 vs 机器漂移」的直接判别**。
#
# 为什么需要判别漂移：§26.48.4 用 CIFAR 已有数据做过一次免费检验——FOLoRA 与 EWC 的
#   seed 5-9 是**同一夜交替跑**的，EWC 持平（+0.13）而 FOLoRA 掉 0.90（INR 上掉 1.10，
#   Welch p≈0.03）。这指向「扩 n 会系统性拉低我方均值、基线不动」，是**平价定位的实证
#   支撑**；但那只是跨配置类比，本批在 INR 上直接测一次才算闭环。
#
# tag 与 CIFAR 侧同名（`ewc_lam300` / `ewc_lam1000`），沿用既有 tag 以免主表拼接错行。
# 代价：**必须显式带 benchmark=imagenetr**（见 BENCHMARK_BY_BATCH 处的坑），否则
# `is_finished` 会拿 cifar100 拼目录，把已完成的 CIFAR run 判成「INR 也跑完了」而静默漏跑。
# ============================================================================
EWC_INR_TOPUP_CONFIGS = [
    ("ewc", "ewc_lam1000", ["--ewc_lambda", "1000"], (5, 6, 7, 8, 9)),
    ("ewc", "ewc_lam300",  ["--ewc_lambda", "300"],  (5, 6, 7, 8, 9)),
    # λ=100 的 tag 是**空串**（目录名由 paths.py 的 `config.tag or "default"` 兜成
    # `default`）。这里必须照抄空串而不是写 "default"，否则新 run 的 config.json 里
    # tag 会与 seed 0-4 的 '' 不一致，按原始 tag 分组的聚合脚本会把同一行拆成两半。
    ("ewc", "",            ["--ewc_lambda", "100"],  (5, 6, 7, 8, 9)),
]

# ============================================================================
# 2026-10-01 收尾批（用户指令：「跑 CCF-C 的推荐剩余实验，第三数据集先不跑」）
# 四批共 24 run，全部在**结果产生之前**冻结（见 FOLoRA决策日志.md §26.41.7 / §26.43 / §26.44）。
# 排序理由、成本实测与「不做清单」都在 §26.44；此处只记每批的**存在理由**，防止
# 后来人（包括我自己）在结果出来之后重新解释这批 run 是干什么的。
# ============================================================================

# ---------------------------------------------- 收尾A：INR FOLoRA 补 seed 到 n=10（5 run，~3.3h）
# 补的是**本项目唯一一个 p<0.05 的跨方法胜**的样本量：INR 主行 `v2f_l3_k64` 66.92 (n=5) vs
# EWC 协议行 λ=300 65.92 (n=5)，配对 **Δ=+1.00，p=0.0091，逐 seed 全正**（§26.22）。
# 它同时也是**最脆**的一条：对 EWC λ=1000（66.02，INR 上均值反而更高那行）只有 +0.90
# p=0.0697。n=5→10 会把这颗胜负的置信区间收窄——**可能变强，也可能被稀释到不显著**。
#
# **必须如实报**：若 n=10 后 p 越过 0.05，就按 §26.39 的「主张降级」把 INR 从
# 「显著领先」改写成「同水平且方差更小」——**不得**回退成 n=5 的旧口径、不得改报 λ。
# 这正是这条 run 的价值：把一个"可能是运气"的胜，变成一个"知道是不是运气"的数。
#
# tag 与 CIFAR 侧**故意同名**（`v2f_l3_k64`）——主表就是靠同名表达"同一配置迁移"。
# 代价是**必须显式带 benchmark=imagenetr**（见 BENCHMARK_BY_BATCH 与该处注释的坑）。
TRANSFER_EXT_CONFIGS = [
    ("folora_v2", "v2f_l3_k64",
     ["--folora_lambda", "3", "--folora_topk", "64"], (5, 6, 7, 8, 9)),
]

# ---------------------------------------------- 收尾B：INR 的 CODA 忠实配置 @20ep（3 run，~6.3h）
# **补的是「论文陈述与实验事实不符」**，与已修过的三处同类（EWC λ 调小、O-LoRA 弱变体、
# CODA 换几何），但这次是**INR 侧从未跑过**：
#   - 论文 setup 明说 prompt 方法**两个基准都训 20 epoch**（05_experiments.tex:150-153, 223-224）；
#   - 主表**两个基准都有 CODA-Prompt 行**（同文件 :238）；
#   - CIFAR 侧由批次 12 补齐了忠实配置（`coda/pool100_len8_ep20`，74.87，n=3）；
#   - **INR 侧只有两个都不能用的候选**：`coda/default`（pool100/len8 但是 **5 epoch**，
#     违反 setup 的 20-epoch 声明）与 `coda/pilot20`（20 epoch 但是 **L2P 的 pool20/len5 几何**，
#     违反"忠实复现 CODA"）。
# ⇒ 不补这一行，INR 的 CODA 格子**无论填哪个都是错的**，而审稿人只要对着 setup 数一遍
#   目录就会发现。这属于「论文对自己实验的描述是错的」，比「方法弱」危险得多。
#
# **tag 故意与 CIFAR 侧不同名**（`pool100_len8_ep20_inr`，CIFAR 侧叫 `pool100_len8_ep20`）：
# 理由是纵深防御——同名 tag 在漏传 benchmark 时会让 `is_finished` **静默查错目录**并把
# 未跑的 run 判成已完成（正是 §26.25.1 第 5 条记录的坑）；`inflora` 的 `ep20`/`ep20_inr`
# 已用同一手法。代价只是表里多一个后缀，而 CIFAR 侧的 `pool100_len8_ep20` 已跑完，
# 不会被本批的命名规则影响。
CODA_FAITHFUL_INR_CONFIGS = [
    ("coda", "pool100_len8_ep20_inr",
     ["--prompt_pool_size", "100", "--prompt_length", "8"], (0, 1, 2)),
]

# ---------------------------------------------- 收尾C：k 消融补到主行 λ=3（10 run，~8.2h）
# 消掉一个**报告口径问题**：现行 k 消融（批次 8）跑在 **λ=10**（`v2f_l10_k16` / `v2f_l10_k128`），
# 而 CIFAR 预注册主行是 **λ=3**（§13.3 规则 2）——本文件 §194 当初的预注册原文写的是
# 「若主网格 5 seeds 最终选出 λ=3，再补 k=16/128 在 λ=3 的 10 个 run」，**这个条件已满足
# 但从未执行**。于是论文若把 k 消融当作主行的消融，就是在用另一个 λ 上的结果支撑主行结论。
#
# **不砍的理由**：它是"我们说过要做但没做"的公开承诺（注释即预注册），审稿人读代码可见。
# **买不到新结论**：λ=10 上 k 已不敏感（k=16 77.90 / k=64 78.24 / k=128 77.30，Δ 均不显著），
# λ=3 上大概率同样不敏感——它的价值是**让"k 不敏感"这句话在正确的 λ 上说**，以及
# 让 memory-accuracy Pareto 前沿（k=16 47.8MB vs k=64 182.8MB）落在主行配置上。
KDIM_L3_CONFIGS = [
    ("folora_v2", "v2f_l3_k16",
     ["--folora_lambda", "3", "--folora_topk", "16"], SEEDS),
    ("folora_v2", "v2f_l3_k128",
     ["--folora_lambda", "3", "--folora_topk", "128"], SEEDS),
]

# ---------------------------------------------- 收尾D：主表 prompt 行补 seed 到 n=5（CIFAR 4 + INR 2 run，~14.5h）
# **为什么是这两行、而不是 `default` 行**（这里曾经写错过一次，见 §26.43.1）：
# 论文 setup 说 prompt 方法训 **20 epoch**，而 `run_prompt_baselines.py` 用的是 `--epochs 5`
# ⇒ **主表该报的是 20-epoch 行**：L2P = `pilot20`（pool20/len5/topk5 @20ep）、
# CODA = `pool100_len8_ep20`（CODA 自己的 pool100/len8 @20ep）。
# `coda/pilot20` 是**几何与 epochs 同时变的混淆配置**，不得当作 CODA 报（日志 §26.x）。
# `default` 行只作「5 epoch vs 20 epoch」的对照留在附录 ⇒ **补 default 的 seed 是白花钱**。
#
# **增益是有条件的**：论文 setup 现写「prompt-based methods ... (3 seeds)」，所以 n=3
# **与陈述自洽**，本批不是"陈述不符"类。它的价值有二：
#   ① 把与 L2P/CODA 的显著性结论从 n=3 加固到 n=5。这是论文里**第二硬的卖点**
#      （CIFAR 上对 L2P/CODA/InfLoRA/O-LoRA 全显著），而它目前建立在 3 颗种子上；
#   ② 补完后若要把 setup 那句改成「5 seeds」，数据是现成的。
# 反过来说，**标题里若只写"prompt 方法 3 seeds"就无需本批**——所以它排在四批的最后，
# 时间不够就从这里砍（见 §26.44 的砍单顺序）。
PROMPT20_TOPUP_CONFIGS = [
    ("l2p", "pilot20",
     ["--prompt_pool_size", "20", "--prompt_length", "5", "--prompt_topk", "5"], (3, 4)),
    ("coda", "pool100_len8_ep20",
     ["--prompt_pool_size", "100", "--prompt_length", "8"], (3, 4)),
]
# INR 侧同理：`l2p/pilot20` 已有 seeds 0-2（20 epoch，与 setup 一致），补 3、4 到 n=5。
# **不建 INR 的 `coda/pool100_len8_ep20` 对等行**——那一行已由本批的 `coda_faithful_inr`
# 以 `pool100_len8_ep20_inr` 的名义负责（故意不同名），避免同一配置出现两个 tag。
PROMPT20_TOPUP_INR_CONFIGS = [
    ("l2p", "pilot20",
     ["--prompt_pool_size", "20", "--prompt_length", "5", "--prompt_topk", "5"], (3, 4)),
]

# 各 batch 的任务数。T=50 的 run 落在 experiments/cifar100/ 下的独立 tag 目录，
# 因此 `run_dir_of` / `is_finished` 不需要知道任务数；只有训练命令需要。
# INR 也是 20 任务（200 类 / 每任务 10 类），与现存 INR run 的 config.json 一致。
NUM_TASKS_BY_BATCH = {"main": 20, "tail": 20, "kdim": 20, "unweighted": 20,
                      "robust": 20, "olora_agg": 20, "olora_orth": 20,
                      "olora_orth_inr": 20, "inflora_faithful": 20,
                      "inflora_faithful_inr": 20,
                      "scaling": 50, "bound": 20, "transfer": 20,
                      "main_ext": 20, "ewc300": 20, "coda_faithful": 20,
                      # 2026-10-01 收尾批（全部 20 任务；INR 同为 200 类 / 每任务 10 类）
                      "transfer_ext": 20, "coda_faithful_inr": 20,
                      "kdim_l3": 20, "prompt20_topup": 20,
                      "prompt20_topup_inr": 20,
                      # 2026-10-01 3n：EWC λ 曲线补到同 n=10（cifar100）
                      "ewc_curve_topup": 20,
                      # 2026-10-01 收尾H：EWC λ 曲线补到同 n=10（imagenetr）
                      "ewc_inr_topup": 20}

MAX_ATTEMPTS = 6      # 单个 run 连续失败多少次后放弃（避免死循环）
RETRY_SLEEP = 60      # 失败后等待秒数（给 OOM/休眠留恢复时间）


def run_dir_of(method: str, tag: str, seed: int,
               benchmark: str = DEFAULT_BENCHMARK) -> Path:
    """`tag` 为空串时目录名是 `default`——与 `src/peft_cl/utils/paths.py` 的
    `label = config.tag or "default"` 必须一致。

    2026-10-01 修：此前直接拼空串会得到 `experiments/<bench>/<method>/seed<k>`，
    与真实目录 `.../<method>/default/seed<k>` 不符 ⇒ `is_finished` 永远返回 False
    ⇒ 该 run 每轮都被判为「没跑」而**无限重跑**。此前没有 tag 为空的批，所以是死代码；
    收尾批 `ewc_inr_topup` 要补 `ewc/default` 的 seed 5-9，才踩到。
    """
    return Path("experiments") / benchmark / method / (tag or "default") / f"seed{seed}"


def is_finished(method: str, tag: str, seed: int,
                benchmark: str = DEFAULT_BENCHMARK) -> bool:
    """默认查 cifar100；查 INR 的 run 必须显式传 benchmark=benchmark_of("transfer")。

    默认值故意保持 cifar100 而不是「按 tag 猜」：`v2f_l3_k64` 在**两个基准上同名**
    （这正是迁移实验想表达的「同一配置」），靠 tag 判断基准一定出错。
    """
    rj = run_dir_of(method, tag, seed, benchmark) / "results.json"
    if not rj.exists():
        return False
    try:
        return bool(json.loads(rj.read_text(encoding="utf-8")).get("finished"))
    except Exception:
        return False


def interleaved_runs(batch: str = "main"):
    """按 seed 轮次交错展开成执行序列。

    轮次取自**该批所有 config 的 seed 并集**，而不是全局 SEEDS：`main_ext` 的种子是
    (5..9)，一个都不在 SEEDS=(0..4) 里，若照旧遍历 SEEDS 会**一个 run 都不产出**——
    而且不报错，`--status` 会打印「0/10 待跑」之后安静地什么都不做，等于这个批次
    静默消失。对现存的每个批，各 config 的 seed 都是 SEEDS 的子集，所以并集恒等于
    SEEDS、产出序列逐项不变（已用 --status 对全部 12 个批核对过）。
    """
    configs = {"main": CONFIGS, "unweighted": UNWEIGHTED_CONFIGS,
               "robust": ROBUSTNESS_CONFIGS, "tail": TAIL_CONFIGS,
               "kdim": KDIM_CONFIGS, "olora_agg": OLORA_AGG_CONFIGS,
               "olora_orth": OLORA_ORTH_CONFIGS,
               "olora_orth_inr": OLORA_ORTH_INR_CONFIGS,
               "inflora_faithful": INFLORA_FAITHFUL_CONFIGS,
               "inflora_faithful_inr": INFLORA_FAITHFUL_INR_CONFIGS,
               "scaling": SCALING_CONFIGS, "bound": BOUND_CONFIGS,
               "transfer": TRANSFER_CONFIGS, "main_ext": MAIN_EXT_CONFIGS,
               "ewc300": EWC300_CONFIGS,
               "coda_faithful": CODA_FAITHFUL_CONFIGS,
               "transfer_ext": TRANSFER_EXT_CONFIGS,
               "coda_faithful_inr": CODA_FAITHFUL_INR_CONFIGS,
               "kdim_l3": KDIM_L3_CONFIGS,
               "prompt20_topup": PROMPT20_TOPUP_CONFIGS,
               "prompt20_topup_inr": PROMPT20_TOPUP_INR_CONFIGS,
               # 2026-10-01 3n（用户拍板「遵守规则 + 补 3n 再报」，见 §26.45）
               "ewc_curve_topup": EWC_CURVE_TOPUP_CONFIGS,
               # 2026-10-01 收尾H（用户拍板「全补」，见 §26.48）
               "ewc_inr_topup": EWC_INR_TOPUP_CONFIGS}[batch]
    rounds = sorted({s for _m, _t, _e, seeds in configs for s in seeds})
    for seed in rounds:
        for method, tag, extra, seeds in configs:
            if seed in seeds:
                yield method, tag, extra, seed


def eval_pending(benchmark: str = DEFAULT_BENCHMARK) -> None:
    """把已完成但没评估过的 run 立刻跑 NCM 评估（幂等：缓存命中瞬间跳过）。

    为什么必须放在这里：`supervise_all` 的 stage 3 是**阻塞**调用本网格的，所以排在它
    后面的 stage 3b（NCM 评估）要等**全部 23 个 run** 跑完才会执行——λ 的第一个可信数字
    （5 seeds 均值）会拖到约 16 小时后才出现，中间十几个小时完全看不到任何结果。
    在每个 run 结束后就地评估，额外代价只有一次目录扫描，换来的是每 ~50 分钟
    就多一个能立刻判断 FOLoRA 是否真的强于 EWC 的 NCM 数字。

    评估失败不能影响训练队列（训练才是稀缺资源），所以整体包 try。
    """
    try:
        subprocess.run([sys.executable, "-u", "-m", "scripts.eval_ncm_sweep",
                        "--benchmark", benchmark, "--num_workers", "4",
                        "--out", f"reports/ncm_summary_{benchmark}.json"])
    except Exception as e:
        print(f"评估失败（不影响训练队列）：{type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="NCM 协议下的 λ 网格（幂等、断点续训友好）")
    ap.add_argument("--batch", default="main",
                    choices=["main", "tail", "kdim", "unweighted", "robust",
                             "olora_agg", "olora_orth", "olora_orth_inr",
                             "inflora_faithful", "inflora_faithful_inr",
                             "scaling", "bound",
                             "transfer", "main_ext", "ewc300", "coda_faithful",
                             "transfer_ext", "coda_faithful_inr", "kdim_l3",
                             "prompt20_topup", "prompt20_topup_inr",
                             "ewc_curve_topup", "ewc_inr_topup"],
                    help="main=论文核心 λ 重调网格；tail=λ=0 锚行；kdim=k 消融；"
                         "unweighted=等权消融；robust=λ 鲁棒性（已降级）；"
                         "olora_agg=O-LoRA 聚合公平性对照；"
                         "olora_orth=O-LoRA 正交性约束（忠实版，补上原文训练期正则项）；"
                         "olora_orth_inr=同上的 ImageNet-R 版（跑在 imagenetr 上，"
                         "修两表 O-LoRA 不同变体的矛盾）；"
                         "scaling=T=50 规模轴（决定性）；"
                         "bound=界的实证探针（1 run，兑现 04_theory.tex 的两条承诺）；"
                         "transfer=ImageNet-R 冻结点迁移（跑在 imagenetr 上，非 cifar100）；"
                         "main_ext=主表两行补 seed 5-9（S-12，把「10 seeds」这句话变成真的）；"
                         "ewc300=EWC λ=300 补齐到 5 seeds（关掉最后一个「基线可能选弱」口子）；"
                         "coda_faithful=CODA 的忠实配置（pool100/len8 @ 20 epoch，最低优先级）；"
                         "inflora_faithful=InfLoRA 训练预算对照（20 epoch，CIFAR）；"
                         "inflora_faithful_inr=同上的 ImageNet-R 版（跑在 imagenetr 上）")
    ap.add_argument("--benchmark", default=None,
                    help="覆盖该 batch 的默认基准（默认由 benchmark_of(batch) 决定："
                         "transfer→imagenetr，其余→cifar100）。")
    ap.add_argument("--status", action="store_true",
                    help="只打印该批次的进度（已完成/待跑），**不启动任何训练**。"
                         "没有这个开关时，任何调用都会真的开跑——包括只想看进度的调用，"
                         "那会和监督器在同一 run 目录上并发写 checkpoint。")
    args = ap.parse_args()
    num_tasks = NUM_TASKS_BY_BATCH[args.batch]
    benchmark = args.benchmark or benchmark_of(args.batch)
    epochs = epochs_of(args.batch)
    runs = list(interleaved_runs(args.batch))
    total = len(runs)

    if args.status:
        done = [(m, t, s) for m, t, _e, s in runs if is_finished(m, t, s, benchmark)]
        todo = [(m, t, s) for m, t, _e, s in runs if not is_finished(m, t, s, benchmark)]
        print(f"batch={args.batch}  benchmark={benchmark}  num_tasks={num_tasks}  "
              f"epochs={epochs}  共 {total} 个 run（已完成 {len(done)}，待跑 {len(todo)}）")
        for m, t, s in todo:
            print(f"  待跑  {m}/{t} seed{s}")
        return

    print(f"batch={args.batch}  benchmark={benchmark}  num_tasks={num_tasks}  "
          f"epochs={epochs}  共 {total} 个 run", flush=True)
    for idx, (method, tag, extra, seed) in enumerate(runs, 1):
        if is_finished(method, tag, seed, benchmark):
            print(f"[{idx}/{total}] 跳过（已完成）{method}/{tag} seed{seed}", flush=True)
            continue
        cmd = [
            sys.executable, "-u", "-m", "scripts.run_single",
            "--benchmark", benchmark, "--num_tasks", str(num_tasks),
            "--method", method, "--seed", str(seed), "--epochs", str(epochs),
            "--lora_rank", str(RANK), "--tag", tag, *extra, "--resume",
        ]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if is_finished(method, tag, seed, benchmark):
                break
            print(f"[{idx}/{total}] 启动 {method}/{tag} seed{seed} "
                  f"(第 {attempt}/{MAX_ATTEMPTS} 次尝试)", flush=True)
            started = time.time()
            proc = subprocess.run(cmd)
            dt = time.time() - started
            if is_finished(method, tag, seed, benchmark):
                print(f"[{idx}/{total}] 完成 {method}/{tag} seed{seed}，"
                      f"用时 {dt/3600:.2f} h", flush=True)
                # 就地评估，让结论尽早可见（理由见 eval_pending docstring）
                eval_pending(benchmark)
                break
            print(f"[{idx}/{total}] 未完成（退出码 {proc.returncode}，{dt/60:.1f} min），"
                  f"{RETRY_SLEEP}s 后续训", flush=True)
            time.sleep(RETRY_SLEEP)
        else:
            print(f"[{idx}/{total}] !! 放弃 {method}/{tag} seed{seed}（重试上限）", flush=True)

    remaining = [(m, t, s) for m, t, _, s in runs
                 if not is_finished(m, t, s, benchmark)]
    print(f"\nλ 网格队列结束。未完成: {remaining if remaining else '无，全部完成'}",
          flush=True)


if __name__ == "__main__":
    main()
