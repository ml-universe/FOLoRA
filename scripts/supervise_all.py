"""总守护：断电 / 重启后一条命令恢复全部未完成实验（幂等、可反复重启）。

为什么需要它
------------
「断电续训」只解决了**单个 run 内部**不丢任务（trainer 每个任务原子落盘 checkpoint）。
但一次实验链条里还有三层会丢：

1. **链条本身**。`nohup bash -c 'a; b; c'` 里 a 死了，b、c 永远不会跑——没有任何
   东西会把链条接上；
2. **单个 run 被 OOM/休眠杀掉**。`run_single` 返回后调用方继续下一个，被杀的那个
   没 finished，没人回头补；
3. **评估缓存的原子性**。NCM 评估靠「缓存 JSON 存在 = 该 run 已完成」来判断，
   写盘途中断电会留下截断的 JSON——旧代码直接 `json.loads` 会**崩掉整个队列**，
   后面的实验全不跑（已修：见 `peft_cl/utils/io.py`）。

本守护把「该做什么」显式声明成 STAGES（按优先级排序），反复循环执行直到每阶段的
pending 都归零。任一阶段崩溃/被杀都不影响其它阶段，重启本守护即可从断点继续——
所有步骤都是幂等的（跳过已完成、`--resume` 续训未完成）。

用法（项目根目录，detached 启动，日志追加）：
  python -u -m scripts.supervise_all >> reports/supervise_all.log 2>&1

重启机器后，重复上面这一条命令即可；或双击 scripts\\resume_all.cmd。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from peft_cl.utils.io import read_json_or_none
from scripts import eval_ncm_sweep as sweep

PY = sys.executable
NUM_WORKERS = "4"
LOCK_PATH = Path("reports/.supervise_all.lock")


def acquire_singleton_lock():
    """独占文件锁：防止同时跑起两个守护。

    两个守护会各自对**同一个 run 目录**调 run_single，交错写同一个 checkpoint，
    结果是两边都以为自己完成了、实际结果错乱——比崩掉更糟，因为它不报错。

    用内核级字节锁而不是「pid 文件 + 存活检查」：Windows 上进程终止（含断电/被杀）
    时锁由内核自动释放，所以不会留下需要手工清理的死锁；而已完成的实验由各阶段
    自己的幂等判断跳过，不需要锁来记录进度。

    返回持有锁的文件对象（必须保持存活），拿不到锁则返回 None。
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCK_PATH, "a+")
    try:
        import msvcrt                       # Windows
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:                     # 非 Windows：退回 fcntl
        import fcntl
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return None
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(f"{os.getpid()} 启动于 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.flush()
    return f


# ---------------------------------------------------------------- 通用工具

def run_finished(bench: str, method: str, tag: str, seed: int) -> bool:
    p = Path("experiments") / bench / method / tag / f"seed{seed}" / "results.json"
    d = read_json_or_none(p)
    return bool(d and d.get("finished"))


def run_single_cmd(bench, method, tag, seed, num_tasks, epochs, extra=()):
    return [PY, "-u", "-m", "scripts.run_single", "--benchmark", bench,
            "--num_tasks", str(num_tasks), "--method", method, "--seed", str(seed),
            "--epochs", str(epochs), "--tag", tag, *extra, "--resume"]


# ---------------------------------------------------------------- 阶段定义

# 阶段 2：训练中途被我杀掉/隔离开的 run（resume 安全，重跑无损）
# 注意 tag="" 表示默认 tag，落盘目录是 experiments/<bench>/<method>/default/
REPAIR_RUNS = [
    ("cifar100", "olora", "default", 2, 20, 5, ()),
    ("cifar100", "olora", "default", 3, 20, 5, ()),
]


def pending_repair():
    return [r for r in REPAIR_RUNS if not run_finished(*r[:4])]


def do_repair():
    for bench, method, tag, seed, num_tasks, epochs, extra in pending_repair():
        print(f"SUPERVISOR: 补齐 {bench}/{method}/{tag} seed{seed}", flush=True)
        subprocess.run(run_single_cmd(bench, method, tag, seed, num_tasks, epochs, extra))


# 阶段 3：NCM 协议下的 λ 重调网格（自带幂等循环）
def pending_lambda_grid():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs() if not is_finished(r[0], r[1], r[3])]


def do_lambda_grid():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid"])


# 等权消融网格（对应 STAGES 里的 3e/3f；**已提升为高优先级**）
# 补的是 Highlights 第 2 条「importance-weighted 优于 equal-weight」——目前无实验支撑。
# 它同时是「主结果只是追平 EWC」时论文唯一的机制级贡献，所以优先级高于 k 消融与 InfLoRA。
def pending_unweighted():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("unweighted") if not is_finished(r[0], r[1], r[3])]


def do_unweighted():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "unweighted"])


# 阶段 8：λ 鲁棒性补充网格（低优先级；**不在关键路径上，可随时砍**）
def pending_robustness():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("robust") if not is_finished(r[0], r[1], r[3])]


def do_robustness():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "robust"])


# 阶段 3c：λ=0 锚行（只剩 5 个 run，排在 InfLoRA 之前只因为便宜）
# **2026-09-23 修正**：本阶段原定为「λ 尾部延伸 λ∈{0.1,0.3,0}，判决正则项是否有用」，
# 依据是「λ 曲线单调上升、拐点未现」。该依据已被推翻——它来自 `v2_l1_k64`=79.65 等
# **10 任务 / 2 epoch / rank 8** 的非主协议 run（详见 run_lambda_grid.py 顶部警告）。
# 主协议（20/5/16）下 λ 曲线**有拐点**：k=16 峰值在 λ=10~30，k=64 上 λ=1(75.45) 已明显
# 低于 λ=10(78.55)。所以 λ=0.1/0.3 已取消，只剩 λ=0 一个档。
#
# λ=0 现在的用途是**消融表的锚行 + 排除一个方法论隐患**：`after_task` 估 Fisher 时多跑的
# forward/backward 若动到 dropout/LayerNorm 的 RNG，λ=0 就不会精确等于 Seq-LoRA，
# 那么「加正则 vs 不加」的对比就被混淆了。落回 Seq-LoRA（n=10 = 69.31±1.92）的噪声带内即排除。
def pending_tail():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("tail") if not is_finished(r[0], r[1], r[3])]


def do_tail():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "tail"])


# k 消融网格（对应 STAGES 里的 8/9，中优先级）
# 补 k 的多 seed 证据 + 出「精度 vs 重要性核显存」的 Pareto 曲线。
def pending_kdim():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("kdim") if not is_finished(r[0], r[1], r[3])]


def do_kdim():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "kdim"])


# 阶段 3g：O-LoRA 聚合公平性对照（3 run，~2.4h）
# O-LoRA 原式把 T 个 adapter 相加，20 任务下 ΔW 秩上限 320/768≈42%，NCM 只有 64.79，
# 低于两条地板线——审稿人会质疑复现坏了。补 mean 聚合对照，两个走向都能用数据回答
# （详见 run_lambda_grid.OLORA_AGG_CONFIGS 的注释）。便宜且堵住一个明确的质疑点。
def pending_olora_agg():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("olora_agg") if not is_finished(r[0], r[1], r[3])]


def do_olora_agg():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "olora_agg"])


# O-LoRA 正交性约束（忠实版，见 run_lambda_grid.OLORA_ORTH_CONFIGS 的注释）。
# 与 3g 相邻：两者都决定主表里 O-LoRA 那一行怎么写（3g 管「怎么合并」，
# 这里管「训练期到底约不约束正交性」）。排在 3h 之后、InfLoRA 之前是因为它更便宜
# （6 run ~4.8h vs ~6h），且直接关系到「基线是否被做弱」这个必然被问的点。
def pending_olora_orth():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("olora_orth")
            if not is_finished(r[0], r[1], r[3])]


def do_olora_orth():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "olora_orth"])


# 阶段 7b：**INR 上的** O-LoRA 正交性约束（2026-09-27 用户拍板「补」，6 run，~3.6h）。
# 补的是两表不自洽：3j 的 `olora_orth` 批只在 cifar100 上跑，于是 CIFAR 表有四个
# O-LoRA 变体、INR 表只有最弱的 `default`——并排看两张表就能发现的矛盾。
# **必须显式传 benchmark**：`olora_orth_l0.1` / `olora_orth_l1` 这两个 tag
# **在两个基准上同名**（CIFAR 版已跑完 n=3），不传就会去查 CIFAR 目录，
# 把 3 个已完成的 CIFAR run 判成「INR 也跑完了」而静默漏跑全部 6 个 INR run
# ——错的不是崩溃，是「看起来正常」。这与 5b 的坑是同一个。
def pending_olora_orth_inr():
    from scripts.run_lambda_grid import benchmark_of, interleaved_runs, is_finished
    bench = benchmark_of("olora_orth_inr")
    return [r for r in interleaved_runs("olora_orth_inr")
            if not is_finished(r[0], r[1], r[3], bench)]


def do_olora_orth_inr():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid",
                    "--batch", "olora_orth_inr"])


# 阶段 7c / 7d：InfLoRA「训练预算对照」（2026-09-28 用户拍板「补」，CIFAR 3 + INR 3 = 6 run，~14h）。
# 补的是**基线复现不可信**：CIFAR `inflora/default` n=5 = 70.14，对同 seed 地板只有 +0.81
# （p=0.351，5 个里 2 个负）；INR seed0 = 50.03，**低于 INR 地板 56.58 达 −6.55**。
# 只改训练预算（5→20 epoch），其余不动；论文里写成 training-budget sensitivity control，
# **不声称官方复现**（无凭据）。预注册与实际代价的拆解见 `run_lambda_grid.INFLORA_FAITHFUL_CONFIGS`
# 上方的长注释。7d 与 7c 的 tag 不同名（`ep20` / `ep20_inr`）⇒ 即使漏传 benchmark 也不会
# 静默查错目录，但**仍然显式传**（纵深防御，同 7b 的教训）。
def pending_inflora_faithful():
    from scripts.run_lambda_grid import benchmark_of, interleaved_runs, is_finished
    bench = benchmark_of("inflora_faithful")
    return [r for r in interleaved_runs("inflora_faithful")
            if not is_finished(r[0], r[1], r[3], bench)]


def do_inflora_faithful():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid",
                    "--batch", "inflora_faithful"])


def pending_inflora_faithful_inr():
    from scripts.run_lambda_grid import benchmark_of, interleaved_runs, is_finished
    bench = benchmark_of("inflora_faithful_inr")
    return [r for r in interleaved_runs("inflora_faithful_inr")
            if not is_finished(r[0], r[1], r[3], bench)]


def do_inflora_faithful_inr():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid",
                    "--batch", "inflora_faithful_inr"])


# T=50 规模轴（对应 STAGES 里的 6/7，**决定性实验**，30 run，~30h）
# 唯一能把「+0.5 点」变成结构性声明的实验：FOLoRA 的正则跨任务累积受保护子空间，
# 任务越多方向相关性越关键，而对角 Fisher 会饱和。若差距随 T 增长，论文就有了
# 「优势随任务数放大」这一条。排在所有**必需项**之后（而不是最前）：万一出意外，
# 先保证有一篇完整的稿子；但它排在 kdim / 鲁棒性之前。
def pending_scaling():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("scaling") if not is_finished(r[0], r[1], r[3])]


def do_scaling():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "scaling"])


# 阶段 3i：界的实证探针（1 run，~1.2h）
# 04_theory.tex 承诺在 Sec. discussion 里实证「高阶项可忽略」与「Assumption 1 的违背量级」，
# 而 sec:discussion 里一条都没有（已 grep）。承诺了不做比不说更伤，且这是**理论段唯一的
# 实证支撑**——在结果趋向「平价」的当下，理论是论文最可能单独扛起贡献的部分。
# 必须在训练过程里测（after_task 会覆盖 ref_params，事后从 checkpoint 拿不到位移）。
def pending_bound():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("bound") if not is_finished(r[0], r[1], r[3])]


def do_bound():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "bound"])


# 阶段 4：InfLoRA 基线（自带幂等循环）
def pending_inflora():
    from scripts.run_inflora_queue import RUNS, is_finished
    return [(b, s) for b, _t, s in RUNS if not is_finished(b, s)]


def do_inflora():
    subprocess.run([PY, "-u", "-m", "scripts.run_inflora_queue"])


# 阶段 5b：ImageNet-R 的「冻结点迁移」网格（自带幂等循环，跑在 imagenetr 上）
# 补的是一个**内部矛盾**：主表有 INR 列，而 INR 上现存的 FOLoRA 只有 v2f_l300_k16
# （λ=300/k=16，即 NCM 网格判定为严重过正则、比 EWC 低 3.6 点的那个点），EWC 则是
# 未被重调的 λ=100。详见 run_lambda_grid.TRANSFER_CONFIGS 的注释。
#
# **必须显式传 benchmark**：`is_finished` 的默认基准是 cifar100，而
# `v2f_l3_k64` / `ewc_lam1000` 这两个 tag 在**两个基准上同名**（迁移实验的语义就是
# 「同一配置」）。不传就会去查 CIFAR 目录，把 5 个已完成的 CIFAR run 判成「INR 也跑完了」
# 而静默漏跑全部 10 个 INR run——错的不是崩溃，是「看起来正常」。
def pending_transfer():
    from scripts.run_lambda_grid import benchmark_of, interleaved_runs, is_finished
    bench = benchmark_of("transfer")
    return [r for r in interleaved_runs("transfer")
            if not is_finished(r[0], r[1], r[3], bench)]


def do_transfer():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "transfer"])


# 阶段 3l：主表两行补 seed 5-9（S-12，10 run，~5h）
# 补的是「论文陈述与实验事实不符」：setup 写「CIFAR-100 上 10 seeds」，但 §13.3 规则 2 冻结
# 出的新主配置（v2f_l3_k64）与 EWC λ=1000 都只有 5 seeds（它们来自只在 SEEDS=0-4 上跑的
# λ 网格）。这是审稿人**数一下目录**就能发现的错，而且方向最糟——不是「方法弱」，而是
# 「论文对自己实验的描述是错的」。详见 run_lambda_grid.MAIN_EXT_CONFIGS 的注释。
# 不传 benchmark：它跑的就是 cifar100（`v2f_l3_k64` 在 INR 上的同名 run 属于 transfer 批，
# 由 5b 显式带 benchmark 处理，两者互不干扰）。
def pending_main_ext():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("main_ext")
            if not is_finished(r[0], r[1], r[3])]


def do_main_ext():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "main_ext"])


# 阶段 3n/3o：EWC λ=300 补齐到 5 seeds（5 run，~2.5h，用户 2026-09-24 决定）
# 关掉**最后一个「基线可能被选弱」口子**：λ=300 是全仓库最高的 EWC 数字（78.02）却只有
# 一颗种子（9/04，已移到 reports/legacy_runs/），而在 seed0 上它高于我们主配置的 77.83。
# 选点规则已在 run_lambda_grid.EWC300_CONFIGS 的注释里**预先冻结**（最高均值点、<0.20 点
# 的差不动主表、≥0.20 点就改主表即使对我们不利）。详见该处注释。
# 与 ROBUSTNESS_CONFIGS 里同样存在的 `ewc/ewc_lam300` (seeds 1,2) 是**同一配置**，
# 谁先跑谁生效、另一批会跳过——不冲突，但意味着砍掉阶段 10/11 也不会影响 λ=300 的 n=5。
def pending_ewc300():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("ewc300")
            if not is_finished(r[0], r[1], r[3])]


def do_ewc300():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "ewc300"])


# 阶段 12：CODA-Prompt 的忠实配置（pool=100/len=8 @ 20 epoch，3 run，~8.7h）—— **排最后**。
# 补的是「基线配置被混淆」：仓库里 coda 的两个 tag 同时差两个变量（default = 原论文几何
# 但只 5 epoch；pilot20 = 论文声明的 20 epoch 但用 L2P 的 pool=20/len=5 几何），而论文
# setup 说 prompt 方法训 20 epoch ⇒ 主表取的是 pilot20，也就是把 CODA 的几何换成了 L2P 的。
# 这与 O-LoRA/EWC 那两处同类（把更弱的变体当该方法报出去）。详见 run_lambda_grid.
# CODA_FAITHFUL_CONFIGS 的注释（含实测代价）。
# **为什么排最后**：它不改变任何结论（prompt 基线离我们 10 点量级），只影响「基线是否被
# 公平对待」的叙事。时间不够就从这里往上砍，零损失。
def pending_coda_faithful():
    from scripts.run_lambda_grid import benchmark_of, interleaved_runs, is_finished
    bench = benchmark_of("coda_faithful")
    return [r for r in interleaved_runs("coda_faithful")
            if not is_finished(r[0], r[1], r[3], bench)]


def do_coda_faithful():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "coda_faithful"])


# ==================== 2026-10-01 收尾批（用户指令：「跑 CCF-C 的推荐剩余实验」）====================
# 四批共 24 run。规格在**结果产生之前**冻结（见 FOLoRA决策日志.md §26.41.7 / §26.43 / §26.44），
# 每批的「存在理由」写在 run_lambda_grid.py 对应 CONFIGS 的注释里。
#
# **顺序即优先级（成本-价值比从高到低）**：
#   14a INR 主行补 n=10（3.3h）—— 最便宜，且直击**唯一 p<0.05 的跨方法胜**
#   14b INR 忠实 CODA @20ep（6.3h）—— **「陈述与数据不符」类**，审稿人对着 setup 数目录即可发现
#   14c k 消融补 λ=3（8.2h）—— 履行本仓库 §194 的**预注册承诺**（我们说过要做但没做）
#   14d/14e prompt 主表行补 n=5（9.9h + 4.6h）—— 加固第二硬卖点，**纯加分**
#
# **砍单顺序（时间不够时从后往前砍）：14d/14e → 14c → 14b → 14a。**
# **14b 不建议砍**：砍了会留下一处「INR 的 CODA 格子填 5ep 违反 setup、填 pilot20 违反忠实复现」
# 的死角，而这是审稿人最容易抓的那一类（不是「方法弱」，是「论文对自己实验的描述是错的」）。
#
# **不需要新增 NCM 评估阶段**：supervise_all 每轮会重新求值**所有**阶段的 pending，
# 现有的 `1./3b/3b2/3m/5/9/13` 等评估阶段会在下一轮自动把新 run 扫进去。
def pending_transfer_ext():
    from scripts.run_lambda_grid import benchmark_of, interleaved_runs, is_finished
    bench = benchmark_of("transfer_ext")
    return [r for r in interleaved_runs("transfer_ext")
            if not is_finished(r[0], r[1], r[3], bench)]


def do_transfer_ext():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "transfer_ext"])


def pending_coda_faithful_inr():
    from scripts.run_lambda_grid import benchmark_of, interleaved_runs, is_finished
    bench = benchmark_of("coda_faithful_inr")
    return [r for r in interleaved_runs("coda_faithful_inr")
            if not is_finished(r[0], r[1], r[3], bench)]


def do_coda_faithful_inr():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "coda_faithful_inr"])


def pending_kdim_l3():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("kdim_l3")
            if not is_finished(r[0], r[1], r[3])]


def do_kdim_l3():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "kdim_l3"])


def pending_prompt20_topup():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("prompt20_topup")
            if not is_finished(r[0], r[1], r[3])]


def do_prompt20_topup():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid", "--batch", "prompt20_topup"])


def pending_prompt20_topup_inr():
    from scripts.run_lambda_grid import benchmark_of, interleaved_runs, is_finished
    bench = benchmark_of("prompt20_topup_inr")
    return [r for r in interleaved_runs("prompt20_topup_inr")
            if not is_finished(r[0], r[1], r[3], bench)]


def do_prompt20_topup_inr():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid",
                    "--batch", "prompt20_topup_inr"])


def pending_ewc_curve_topup():
    from scripts.run_lambda_grid import interleaved_runs, is_finished
    return [r for r in interleaved_runs("ewc_curve_topup")
            if not is_finished(r[0], r[1], r[3])]


def do_ewc_curve_topup():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid",
                    "--batch", "ewc_curve_topup"])


def pending_ewc_inr_topup():
    """EWC 的 ImageNet-R λ 曲线补到同 n=10（收尾H）。

    **基准必须显式传**：这批跑在 imagenetr 上。`is_finished` 的默认参数是 cifar100，
    而这三个 tag 在 CIFAR 侧都**已经存在且已完成**（`ewc_lam1000` n=10、`ewc_lam300`
    n=5、`''` n=10）——不显式传 benchmark 的话，这里会把 CIFAR 的完成状态当成 INR 的，
    返回「待办 0」而**静默漏跑整批**。这正是 `benchmark_of` 与 `BENCHMARK_BY_BATCH`
    那两条注释反复警告的坑。
    """
    from scripts.run_lambda_grid import interleaved_runs, is_finished, benchmark_of
    need = benchmark_of("ewc_inr_topup")
    return [r for r in interleaved_runs("ewc_inr_topup")
            if not is_finished(r[0], r[1], r[3], benchmark=need)]


def do_ewc_inr_topup():
    subprocess.run([PY, "-u", "-m", "scripts.run_lambda_grid",
                    "--batch", "ewc_inr_topup"])


def pending_ncm_matrix():
    """主表 run 中还没有矩阵落盘的那些（零训练，只重跑 NCM 评估）。"""
    from pathlib import Path
    from scripts.eval_ncm_matrix import MAIN_TAGS
    from scripts.eval_ncm_sweep import discover_runs
    todo = []
    for bench, tags_map in MAIN_TAGS.items():
        per = {}
        for method, tag, seed, _sd, _d in discover_runs(
                Path("experiments"), bench, list(tags_map)):
            if tag not in tags_map.get(method, []):
                continue
            key = (method, tag)
            if per.get(key, 0) >= 5:          # 与 eval_ncm_matrix 的 --max_seeds 默认一致
                continue
            per[key] = per.get(key, 0) + 1
            out = (Path("reports/ncm_matrix") / bench /
                   f"{method}__{tag}__seed{seed}.json")
            if not out.exists():
                todo.append((bench, method, tag, seed))
    return todo


def do_ncm_matrix():
    for bench in ("cifar100", "imagenetr"):
        subprocess.run([PY, "-u", "-m", "scripts.eval_ncm_matrix",
                        "--benchmark", bench])


# 阶段 1/5：NCM 全量评估（零训练，逐 run 缓存）
def sweep_pending(bench: str) -> int:
    methods = sweep.SWEEP_METHODS
    cache_root = Path("reports/ncm")
    runs = sweep.discover_runs(Path("experiments"), bench, methods)
    n = 0
    for method, tag, seed, _run_dir, _res in runs:
        cp = sweep.cache_path(cache_root, bench, method, tag, seed)
        # 截断的缓存算「未完成」，重算即可（read_json_or_none 返回 None）
        if cp.exists() and read_json_or_none(cp) is not None:
            continue
        n += 1
    # SimpleCIL 冻结特征基线只依赖 benchmark，只评估一次
    if runs and read_json_or_none(sweep.cache_path(cache_root, bench, "simplecil",
                                                   "frozen", 0)) is None:
        n += 1
    return n


def make_do_sweep(bench):
    def _do():
        subprocess.run([PY, "-u", "-m", "scripts.eval_ncm_sweep", "--benchmark", bench,
                        "--num_workers", NUM_WORKERS,
                        "--out", f"reports/ncm_summary_{bench}.json"])
    return _do


# (阶段名, pending 函数, 执行函数) —— **顺序即优先级**。
#
# **2026-09-25 重排**（用户指令：「所有有用的东西都不砍，力保数据真实最优」）。
# **没有任何阶段被删除**，只改执行顺序。触发原因是当轮实测结论变了：
#   CIFAR 上 FOLoRA 对 EWC 的三个 λ 点全部判「平价」（p=0.0734 / 0.4961 / 0.2277），
#   INR 上反而落后 2.34 点（p=0.0836，逐 seed 差 4/5 为负）。
# 精度不再是差异化来源，于是排序目标从「尽早出精度结论」改为：
#   A 组 零成本收尾（幂等，能跑就跑）
#   B 组 **先关掉必然被拒的口子**：声明无证据（3e、3i）、主表自相矛盾（5b）、
#        平价下仅存的差异化证据（10 λ 鲁棒性，§11.5 的复活条件已满足）
#   C/D 组 基线忠实性（3g/3j O-LoRA 低于 seq 地板；4 InfLoRA 引了未比）
#   E 组 **赌注**：6 T=50 —— 唯一可能改变论文档次的实验
#   F 组 厚度项（k 消融、λ=0 锚行、CODA），不影响任何结论
# 判据是**信息量/小时**而不是花费大小：supervise_all 是**串行阻塞**的，
# 一个 30h 的阶段会把后面所有阶段一起挡住，所以长赌注必须排在硬伤之后、厚度之前。
STAGES = [
    ("1. NCM 评估（补齐 CIFAR 基线）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    ("2. 补齐被中断的 O-LoRA", lambda: len(pending_repair()), do_repair),
    ("3. λ 重调网格（论文核心，~17h）", lambda: len(pending_lambda_grid()),
     do_lambda_grid),
    # 3b 必须紧跟 3：它是「λ 网格出结论」的关键路径。若等到下一轮才评估，
    # 论文核心决策会被 InfLoRA 训练（~6h）挡在后面，白等 6 小时。
    # 重复调用无浪费——已缓存的结果会被瞬间跳过。
    ("3b. NCM 评估（λ 网格新 run，关键路径）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    # 3b2：ImageNet-R 的 NCM 评估**提前到这里**（2026-09-24 新增）。三个理由，都不是
    # 「再等等也行」：
    # 1) **它从未执行过**。监督器每次启动都只走到「第 1 轮」就中断，而阶段 3 的 do() 是
    #    阻塞 17h 的整个 λ 网格——所以列表里排在 3 之后的阶段（含原阶段 5 的 INR 评估）
    #    一次都没跑过。INR 的 NCM 缓存目录 reports/ncm/imagenetr/ 根本不存在（0 个文件），
    #    而 reports/ncm/cifar100/ 有 97 个。主表 INR 列的 NCM 数字目前是**空的**。
    # 2) 零训练代价（纯评估已有 checkpoint，32 run ≈ 2h），但它挡着第二基准的整列数字。
    # 3) 它是**没被验证过的代码路径**。花 2h 现在发现「INR 的 checkpoint 加载不了」，
    #    比 10 小时后才发现便宜得多。
    # 重复调用无浪费：与阶段 5 是同一个幂等 sweep，缓存命中会瞬间跳过。
    ("3b2. NCM 评估（ImageNet-R，首次执行，零训练 ~2h）",
     lambda: sweep_pending("imagenetr"), make_do_sweep("imagenetr")),
    # 3i 已下移到 3o 之后（B 组）。此处留位说明移动原因：它当时被排在 3c 之前是因为
    # 「最便宜」，但 3c 本身已不再是决策项（§11.2），所以这个理由失效。见新位置的注释。
    # 3l/3m：主表两行补 seed 5-9（S-12，10 run，~5h）。**2026-09-24 新增**（用户决定）。
    # 为什么插在 3i 之后、3c 之前（而不是更靠后）：
    # 1) 它修的是**主结果表**上的「论文陈述与数据不符」，而主表是审稿人第一个看的东西。
    #    这类错最便宜被发现（数目录），也最伤可信度——「方法只赢 0.5 点」是学术判断，
    #    「说 10 个 seed 实际 5 个」是陈述不实。修它的收益与结果好坏无关，一定兑现。
    # 2) 它在 cifar100 管线里、与阶段 3 同基准同配置，代码路径是热的；留到最后再回来
    #    重碰这条管线，多一次环境风险。
    # 3) 5h 比 3e 等权消融（~7h）、4 InfLoRA（~6h）、5b INR 迁移（~5.5h）都便宜，而
    #    在剩余阶段里它的「最坏情况伤害」最高，所以按性价比排在最前。
    # 4) 它**不 gate 任何决策**（λ 判决来自已冻结的 §13.3 规则），所以绝不能插在
    #    3b/3b2/3i 这些决策/承诺关键路径之前——这正是它排在 3i 之后的原因。
    # 3m 评估不可省：不评估的 run 等于没跑（S-11 就是「run 跑完但从未评估」）。
    ("3l. 主表补 seed 5-9（S-12：把「10 seeds」变成真的，10 run，~5h）",
     lambda: len(pending_main_ext()), do_main_ext),
    ("3m. NCM 评估（补 seed 新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    # 3n/3o：EWC λ=300 补到 n=5（5 run，~2.5h）。为什么排在这里而不是更靠前：
    # 它与 3l/3m 同类（都是「让主表自洽」的补测），而 3l 已先排定；更重要的是它**在
    # λ=1000 补到 5 seeds 之后跑更划算**——那时 λ∈{100,300,1000,3000} 里除 λ=100 外都是
    # n=5，选点规则要求的「同 n 口径比较」才成立。它排在 InfLoRA（4）之前，因为 InfLoRA
    # 是最大拒稿口但也最贵（~6h），而这里 2.5h 就能关掉一个「基线被选弱」的质疑。
    # 详见 run_lambda_grid.EWC300_CONFIGS 的注释（含预先冻结的选点规则）。
    ("3n. EWC λ=300 补齐到 5 seeds（关掉最后一个「基线可能选弱」口子，5 run，~2.5h）",
     lambda: len(pending_ewc300()), do_ewc300),
    ("3o. NCM 评估（EWC λ=300 新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    # 3i：界的实证探针。**2026-09-25 提到 B 组**，因为「承诺了却没数据」和 3e 同类
    # （声明无证据），而现在精度已判平价，理论段是论文最可能单独站住的部分，
    # 它唯一的实证支撑不能是空的。
    # **重跑理由**：上一次执行 19/19 全败，报的是确定性错误
    # `RuntimeError: Expected all tensors to be on the same device, but got mat is on cpu,
    # different from other tensors on cuda:0 ... wrapper_CUDA_addmv_`
    # ——即探针在 CPU 上建了张量而模型在 cuda:0。run 本身「跑完」了（产出一个 NCM=77.06），
    # 但**零成功测量**，所以理论段两条承诺仍然没有数据。`is_finished` 只看 results.json 的
    # `finished` 标志，因此这个失败 run 会被判为已完成、**pending()=0 永不重试**——
    # 必须先删掉该 run 的产物（见本节末尾「3i 重跑操作」），修好 device bug 后它才会重新入队。
    ("3i. 界的实证探针（1 run，~1.2h，兑现理论段两条承诺）",
     lambda: len(pending_bound()), do_bound),
    # 3e/3f：**等权消融提到这里**（原在 8/9）。
    # 排序理由（2026-09-23 调整）：如果主结果只是「精度追平 EWC-LoRA」，那么论文唯一的
    # 机制级贡献就是「σ² 加权的方向保护优于等权保护」——它同时是 Highlights 第 2 条的
    # **唯一支撑**，而那条声明目前零实验证据（被声明却无证据，比"不确定更强"更严重）。
    # λ 网格只能回答「平价还是小胜」，等权消融才能回答「贡献是什么」，所以后者优先。
    # 另外它**必须**排在 k 消融（6/7）之前：k 消融若显示 k=16 更好，等权对照也该在 k=16 上做，
    # 先跑 k 消融可以避免白跑。此比较是**同 k 同 λ 的受控对比**，故 k 取 64 本身不影响结论有效性。
    ("3e. 等权消融网格（Highlights 第 2 条的唯一支撑，~7h）",
     lambda: len(pending_unweighted()), do_unweighted),
    ("3f. NCM 评估（等权消融新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    # 5b/5c 紧跟 3e/3f（2026-09-25 从 D 组之后的原位提到这里）。它修的是主表 INR 列的
    # 内部矛盾：我们的方法跑在 λ=300/k=16——正是论文自己判定「严重过正则」的点，
    # 而 EWC 跑在未重调的 λ=100。这是已修掉的 EWC 调参缺陷的**镜像**（把弱配置报成自己
    # 的成绩 = 把强配置留给基线）。同时它是 §11.3 那条方法论红线（INR 必须沿用 CIFAR
    # 选出的 λ/k、不得重调，否则就是两个基准分别调参）的**唯一执行**——所以它不只是补数字，
    # 是把「INR 算留出验证」这件事做实。5.5h，且是审稿人对照两张表就能发现的矛盾。
    #
    # 注意别把它当成 INR 的免费翻盘：TRANSFER_CONFIGS 对**两个方法都**迁移了冻结点
    # （FOLoRA→λ=3/k=64，EWC→λ=1000），所以这是一场公平赛，结果未知。
    ("5b. INR 冻结点迁移网格（10 run，~5.5h，修主表 INR 列的内部矛盾）",
     lambda: len(pending_transfer()), do_transfer),
    ("5c. NCM 评估（ImageNet-R，迁移 run）", lambda: sweep_pending("imagenetr"),
     make_do_sweep("imagenetr")),

    # 10/11 **从「已降级，可砍」提升到这里**（2026-09-25）。§11.5 当初推翻「λ 鲁棒性比
    # EWC 强 3 倍」时留了明确的复活条件：「**除非主网格给出 FOLoRA 在 λ∈{1,3,10} 的
    # 5-seed 曲线后，能用同一区间重新论证**」。该条件现已满足——λ 网格跑完，FOLoRA 在
    # λ∈[1,10] 上是 77.33(n=5) / 77.87(n=10) / 78.24(n=5)，极差仅 0.91。所以这不是翻
    # 旧账，是兑现 §11.5 自己写下的条件。
    #
    # **但数据还不匹配**：EWC 的 λ=1/λ=10 目前是 n=1，而 §11.0.1 实测「同 seed 同配置重跑
    # 就差 1.3–1.5 点」——单 seed 的极差无论多大都不构成证据。ROBUSTNESS_CONFIGS 已扩到与
    # FOLoRA 主网格**同 seed 的 n=5**，并补上原本缺失的 λ=3（原网格是 1,10,30,…,跳过了 3，
    # 于是「同一区间」里唯一能与 FOLoRA λ=3 对齐的点不存在）。只有匹配 n、匹配 λ 点的
    # 同区间比较才站得住，否则就是把 §11.5 的错误再犯一遍。
    #
    # 排在 T=50（~30h）之前：6.5h 先买下「赌输时的退路」，比在 30h 赌注之后才发现没有
    # 退路便宜得多。若 T=50 赢，这 6.5h 只是 30h 的零头；若 T=50 平，它就是主要卖点。
    ("10. λ 鲁棒性网格（匹配区间论证的唯一支撑，同 seed n=5）",
     lambda: len(pending_robustness()), do_robustness),
    ("11. NCM 评估（鲁棒性网格新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),

    # 3g/3h：O-LoRA 聚合对照。只 3 个 run（~2.4h），却能挡掉一个必然出现的审稿质疑
    # （「你 O-LoRA 只有 64.79，是复现坏了吧」）。直接决定主表里 O-LoRA 那一行怎么写。
    # 2026-09-25 补：该质疑现在**更硬**了——O-LoRA 在两个基准上都低于 seq 地板
    # （CIFAR 64.79 < 69.31，INR 44.79 < 56.58），跨基准一致说明是配置的真实性质
    # 而非偶发，所以不能不解释。
    ("3g. O-LoRA 聚合公平性对照（3 run，~2.4h）", lambda: len(pending_olora_agg()),
     do_olora_agg),
    ("3h. NCM 评估（O-LoRA 对照新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    # 3j/3k：O-LoRA 的正交性**训练期约束**（原文目标里就有的 λ₁ 项）。
    # 为什么必须补：现在主表的 "O-LoRA" 只做正交初始化、`regularization_loss` 恒 0，
    # 即训练期完全不约束——比原方法更弱。把弱变体当 O-LoRA 报，同时用它的低分支撑
    # 论文动机段，是「基线未忠实复现」这个必然质疑的最坏形态（和 EWC λ 调小同类）。
    # 6 run (~4.8h)；若时间不足的退路是**如实降级命名**（写成 "orthogonal-init LoRA"
    # 并说明与原文差别），而不是继续叫它 O-LoRA——见 OLORA_ORTH_CONFIGS 注释。
    ("3j. O-LoRA 正交性约束（忠实版，6 run，~4.8h）", lambda: len(pending_olora_orth()),
     do_olora_orth),
    ("3k. NCM 评估（O-LoRA 正交版新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    ("4. InfLoRA 基线（CIFAR 5 + INR 3，~6h）", lambda: len(pending_inflora()),
     do_inflora),
    ("5. NCM 评估（ImageNet-R）", lambda: sweep_pending("imagenetr"),
     make_do_sweep("imagenetr")),
    # 5b/5c 已上移至 B 组（紧随 3e/3f）——完整理由见那里。此处留位以免被误读为删除。
    # 6/7：T=50 规模轴 —— **整个计划里唯一可能改变论文档次**的实验（30 run，~30h）。
    # 2026-09-25 位置不变（仍在 A–D 组之后），但**理由变了**：原先是对冲「万一机器出事，
    # 先保证一篇完整的稿子」。现在精度已判平价，A–D 组里全是「不做就必然被拒」的项
    # （声明无证据 / 主表自相矛盾 / 基线引了未比），它们才是稿子的下限；T=50 是唯一还可能
    # 把稿子从「平庸」抬到「能看」的实验。所以它仍在硬伤之后，但**提前到所有厚度项之前**。
    # 原注释「排在 k 消融/鲁棒性之前」中的鲁棒性已上调（见 10/11），此处更正。
    # T=50 的 run 由 eval_ncm_sweep 按 config.num_tasks 分桶，不会污染 T=20 主表。
    ("6. T=50 规模轴（决定性实验，30 run，~30h）", lambda: len(pending_scaling()),
     do_scaling),
    ("7. NCM 评估（T=50 新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    # 7b：INR 的 O-LoRA 忠实版（6 run，~3.6h）。**为什么排在 T=50 之后**：
    # 它修的是「两张主表的 O-LoRA 不是同一个变体」这个叙事矛盾，**不 gate 任何数字结论**
    # （不像 5b 那样直接替换主表里我们自己的那一行）；而 T=50 是本计划里**唯一可能改变
    # 论文档次**的实验，它的结果到达时间越早越好。所以 7b 排在 T=50 之后、但在纯厚度项
    # （8 k 消融 / 3c 锚行 / 12 CODA）之前——它仍是"基线忠实性"类，优先级高于厚度项。
    # 若用户要求提前，移到 3k 之后即可（各阶段幂等，顺序不影响总量，只影响谁先出结果）。
    ("7b. INR O-LoRA 正交性约束（忠实版，6 run，~3.6h，修两表 O-LoRA 不同变体的矛盾）",
     lambda: len(pending_olora_orth_inr()), do_olora_orth_inr),
    # 7c/7d：InfLoRA 训练预算对照（20ep，CIFAR 3 + INR 3 = 6 run，~14h）。
    # **为什么排在 7b 之后、厚度项之前**：与 7b 同类（修「基线复现不可信」这个叙事/可信性
    # 硬伤），但 7c/7d 的 INR 那一行是**低于地板**（−6.55）——比 7b 的「不同变体」更硬，
    # 只是它落在 INR 而非主表；T=50 仍是唯一改档次的实验，所以这三批统一排在 T=50 之后。
    # 排在 k 消融/锚行/CODA 之前：那三项是纯厚度（不加也能投），本批是可信性。
    # 顺序理由与 7b 相同：各阶段幂等，先后只影响谁先出结果，不影响总量与总 ETA。
    ("7c. InfLoRA 训练预算对照（20ep，CIFAR，3 run，~8.7h，修 CIFAR 复现贴地板）",
     lambda: len(pending_inflora_faithful()), do_inflora_faithful),
    ("7d. InfLoRA 训练预算对照（20ep，ImageNet-R，3 run，~5.3h，修 INR 复现低于地板）",
     lambda: len(pending_inflora_faithful_inr()), do_inflora_faithful_inr),

    # ---- F 组：厚度项（不影响任何结论，纯加分）----
    # k 消融（8/9）不必排在等权消融之前：等权消融是**同 k 同 λ 的受控对比**
    # （两边只差 `folora_weighted` 一个开关），k 取 64 本身不影响那条对照的有效性。
    # 它还有个副产品：k 是 FOLoRA 显存的唯一驱动因素（§11.1：k=16→47.8MB、k=64→182.8MB，
    # 而 EWC 只有 5.6MB）。在「绝不能主张比 EWC 省内存」的前提下（§11.1），
    # 这条曲线是全文唯一能诚实呈现该代价的地方——所以它不能砍，只是不必早跑。
    ("8. k 消融网格（k∈{16,128} @ λ=10，中优先级）", lambda: len(pending_kdim()),
     do_kdim),
    ("9. NCM 评估（k 消融新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    # 3c/3d **从关键路径移到 F 组**（2026-09-25）。§11.2 已把本阶段从「方法是否成立的
    # 判决」降为「消融表锚行 + 排除 after_task 的 RNG 副作用」——它不 gate 任何结论，
    # 不该占着 3e/5b 之前的决策路径（那里现在放的是会被拒稿的项）。
    ("3c. λ=0 锚行（5 run，~4h）", lambda: len(pending_tail()), do_tail),
    ("3d. NCM 评估（λ=0 新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),
    # 12/13：**仍是最低优先级**（详见 CODA_FAITHFUL_CONFIGS 注释）。用户已明确「不砍」，
    # 故保留全部条目，只排在最后。
    ("12. CODA 忠实配置（pool100/len8 @ 20ep，3 run，~8.7h）",
     lambda: len(pending_coda_faithful()), do_coda_faithful),
    ("13. NCM 评估（CODA 忠实配置新 run）", lambda: sweep_pending("cifar100"),
     make_do_sweep("cifar100")),

    # ---- G 组：2026-10-01 收尾批（CCF-C 推荐剩余实验；顺序即优先级，理由见上方注释块）----
    ("14a. INR 主行补 seed 5-9（n=5→10，5 run，~3.3h）",
     lambda: len(pending_transfer_ext()), do_transfer_ext),
    ("14b. INR 的 CODA 忠实配置 @20ep（3 run，~6.3h，修「陈述与数据不符」）",
     lambda: len(pending_coda_faithful_inr()), do_coda_faithful_inr),
    ("14c. k 消融补 λ=3（k∈{16,128}×5 seeds，10 run，~8.2h，履行预注册）",
     lambda: len(pending_kdim_l3()), do_kdim_l3),
    ("14d. 主表 prompt 行补 seed 3-4（CIFAR，4 run，~9.9h）",
     lambda: len(pending_prompt20_topup()), do_prompt20_topup),
    ("14e. 主表 prompt 行补 seed 3-4（ImageNet-R，2 run，~4.6h）",
     lambda: len(pending_prompt20_topup_inr()), do_prompt20_topup_inr),
    # 14f 排在最后：它只 gate「主表 EWC 报哪一行」，而 P1 重建主表必须等 14d/14e
    # 补完 prompt 行的 n 之后才做 —— 所以放哪儿对 P1 都等价。放尾部以保住上面五批
    # 已定的优先级。若用户想把 ★ 结论提前锁死，可把本项上移（只改顺序，不改规格）。
    ("14f. EWC λ 曲线补到同 n=10（3n，12 run，~9.1h，拍板项）",
     lambda: len(pending_ewc_curve_topup()), do_ewc_curve_topup),
    # 14g 是**纯增值项**：不改任何已有数字，只把 eval_ncm.py 早已算出、却被
    # eval_ncm_sweep.py 丢掉的 20×20 准确率矩阵落盘，用于画 NCM 协议的遗忘曲线
    # （论文目前只有 1 张方法框架图、实验章 0 张图）。**若时间不够，砍掉它不影响
    # 任何结论**——所以放在最后。脚本自带与 reports/ncm 标量的一致性校验，不一致
    # 即说明 NCM 评估不可复现，那时应该停下来查而不是继续累积。
    # 14h = 14f 的 ImageNet-R 镜像，用户 2026-10-01 拍板「全补」。
    # 排 14f 之后：CIFAR 侧的主表行判定（P1 重建主表要用）先落地。
    # 它买入的是**口径自洽 + 「seed 噪声 vs 机器漂移」的直接判别**，不是显著性——
    # §26.48 的预测分布模拟已预先量化：对 λ=1000 的 Δ 预期从 +0.90 落到 +0.35
    # （p 从 0.070 到 ≈0.29），只有 3.1% 的概率反而变显著。两种结果都如实报。
    ("14h. EWC λ 曲线补到同 n=10（INR，15 run，~8.3h，拍板项「全补」）",
     lambda: len(pending_ewc_inr_topup()), do_ewc_inr_topup),
    # 14g 是**纯增值项**：不改任何已有数字，只把 eval_ncm.py 早已算出、却被
    # eval_ncm_sweep.py 丢掉的 20×20 准确率矩阵落盘，用于画 NCM 协议的遗忘曲线
    # （论文目前只有 1 张方法框架图、实验章 0 张图）。**若时间不够，砍掉它不影响
    # 任何结论**——所以放在最后。脚本自带与 reports/ncm 标量的一致性校验，不一致
    # 即说明 NCM 评估不可复现，那时应该停下来查而不是继续累积。
    ("14g. NCM 准确率矩阵落盘（画遗忘曲线用，~58 run，~2-4h，可砍）",
     lambda: len(pending_ncm_matrix()), do_ncm_matrix),
]


def main() -> None:
    lock = acquire_singleton_lock()
    if lock is None:
        print("SUPERVISOR: 已有一个守护在运行，本进程退出（避免两个守护写坏同一批 run）。",
              flush=True)
        return
    print(f"SUPERVISOR: 已获得单例锁 {LOCK_PATH}，pid={os.getpid()}", flush=True)

    round_no = 0
    while True:
        round_no += 1
        todo = []
        for name, pending, do in STAGES:
            try:
                n = pending()
            except Exception as e:
                print(f"SUPERVISOR: 阶段「{name}」pending 检查异常 "
                      f"{type(e).__name__}: {e}", flush=True)
                n = 0
            if n:
                todo.append((name, n, do))
        if not todo:
            print("SUPERVISOR: 全部阶段均已归零，退出。", flush=True)
            return

        print(f"\n===== SUPERVISOR 第 {round_no} 轮 =====", flush=True)
        for name, n, do in todo:
            print(f"SUPERVISOR: 待办 {n:>3} 项 —— {name}", flush=True)
        for name, n, do in todo:
            try:
                do()
            except Exception as e:      # 单个阶段失败不阻断其它阶段
                print(f"SUPERVISOR: 阶段「{name}」异常 {type(e).__name__}: {e}",
                      flush=True)
        # 一轮走完还有剩活 → 说明有阶段失败或被中途杀掉，等一会儿再补一轮
        if any(_safe_pending(pending) for _name, pending, _do in STAGES):
            print("SUPERVISOR: 本轮结束后仍有剩余，30s 后进入下一轮。", flush=True)
            time.sleep(30)


def _safe_pending(pending):
    try:
        return bool(pending())
    except Exception:
        return False


if __name__ == "__main__":
    main()
