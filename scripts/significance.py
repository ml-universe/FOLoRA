"""显著性检验：论文表格里的「FOLoRA 优于基线」是否有统计支撑。

用法：
    python -m scripts.significance                    # 论文主表（NCM 协议 + paired t，默认）
    python -m scripts.significance --benchmark all    # 两个基准一起
    python -m scripts.significance --all-configs      # 扫全部 config（含消融/调优）
    python -m scripts.significance --test welch       # 退回 Welch 独立样本 t，仅供对照
    python -m scripts.significance --check-paper      # 回验论文正文引用的 p 值
    python -m scripts.significance --source results   # 退回旧口径（线性头 CIL），仅供对照
    python -m scripts.significance --family bench     # 改家族划分（见下「多重比较」）

**数据来源（`--source`）**：
- `ncm`（默认）：读 `reports/ncm/{benchmark}/*.json` —— 这是**论文现在用的协议**
  （冻结/适配特征 + 最近类均值）。该缓存由 `scripts/eval_ncm_sweep.py` 生成。
- `results`：读 `experiments/{benchmark}/{method}/{tag}/seed*/results.json` 里的
  `final_acc_cil`，即「随任务增长的可训练线性头」旧口径。那个指标下 CIL 只有 8~11%，
  几乎被分类头主导，**不能用来支持论文声明**，留作对照。

两个来源的 tag 记法不同：`results` 里 default 的 tag 是空串（config 的 `tag` 字段），
而 NCM 缓存里是目录名 `default`。所以主表对照关系分成 `PAPER_MAIN` / `PAPER_MAIN_NCM` 两张。

**口径过滤**：NCM 缓存记录本身不含 `epochs`/`lora_rank`，只有 `num_tasks`，所以协议
分组时回读每条记录 `<run_dir>/config.json` 补齐。必须要求 `num_tasks==20 且 epochs==5
且 lora_rank==16`——**只看 num_tasks 会把 `*/pilot20`（20 任务但 20 epoch）混进来**，
那批精度低 5~10 点，混进去会让结论全错。`results` 口径下 config 就在记录里，直接用。

默认检验口径 = **paired t-test**（配对 t 检验）。各方法的 seed 已由
scripts/run_seed_align.py 对齐，「同 seed 同初始权重」下配对检验功效更高，而且这正是
FOLoRA决策日志.md §13.3 **在 seed 3/4 落地之前就预注册**的判据口径
（paired t on common seeds，基线固定 EWC λ=100）。
`--test welch` 可退回独立样本 Welch t 检验（`equal_var=False`，方差不齐），
但**论文口径是 paired**；换口径必须在正文同步写明。

**多重比较（Holm–Bonferroni）**：主表是「本文方法 vs K 条基线」，K 条基线各测一次
就是 K 个假设；不校正时，「K 次里至少一次误报」的概率随 K 上升（K=6 时约 26%）。
默认对同一 benchmark、同一指标下的全部对照施加 Holm step-down 校正，产物里新增
`p_holm` / `significant_holm` / `family` / `family_size` 四个字段；原始 `p` 保留不删
（正文里那些「本来就不显著」的诚实读数仍按原始 p 讲）。家族怎么划由 `--family` 决定，
**换家族必须在正文同步写明**，因为它是结论的一部分而不是实现细节。
"""

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from scipy import stats

from peft_cl.utils.io import read_json_or_none

NCM_CACHE_ROOT = Path("reports/ncm")
# 论文主表口径：20 任务 × 5 epoch × rank 16。见模块 docstring「口径过滤」。
MAIN_PROTO = (20, 5, 16)
# **epoch 预算按方法族不同**（论文 Setup 里写明了：adapter 类 5 epoch 防过拟合且让经验
# Fisher 有意义，prompt 类 20 epoch 因为 key 需要更多轮）。所以口径过滤不能全局用同一个
# 三元组——那会把 prompt 基线全滤掉，主表直接缺 L2P/CODA 两行。
ADAPTER_METHODS = {"seq", "ewc", "olora", "inflora", "folora", "folora_v2"}


def expected_proto(method: str):
    """该方法族在论文主表里应有的口径。同时充当 pilot 判据：
    `*/pilot20` 对 adapter 类方法是 epochs=20，与应有的 5 不符，会被判掉。"""
    return MAIN_PROTO if method in ADAPTER_METHODS else (20, 20, 16)

# 论文 Table 1 / Table 2 的对照关系：(标签, method, tag)。
# 注意 tag 取 config 里的 tag 字段，不是目录名——目录叫 default 的，tag 通常是空串。
PAPER_MAIN = {
    "cifar100": [
        ("Seq-LoRA", "seq", ""),
        ("EWC-LoRA", "ewc", ""),
        ("O-LoRA", "olora", ""),
        ("L2P", "l2p", "pilot20"),
        ("CODA-Prompt", "coda", "pilot20"),
    ],
    "imagenetr": [
        ("Seq-LoRA", "seq", ""),
        ("EWC-LoRA", "ewc", ""),
        ("O-LoRA", "olora", ""),
        ("L2P", "l2p", "pilot20"),
        ("CODA-Prompt", "coda", "pilot20"),
    ],
}

# 同一张主表在 NCM 口径下的对照关系。注意 tag 是**目录名**（default），
# 而 results 口径里 default 的 tag 是空串——两套记法不能混用。
#
# !! 2026-10-03：本表先前写死「EWC 只取一档」和「CODA 取 pilot20」，两处都在挑配置，
# 已按预注册规则改正。改动理由逐条记在下面，**不要退回旧写法**：
#
# (1) EWC **四档全列**（λ=100/300/1000/3000），判据基线固定 λ=100。
#     旧写法 CIFAR 取 `ewc_lam1000`（注释理由是「调优后的最强 EWC」），但 λ 网格补到
#     同 n=10 后该前提**已假**：CIFAR 最强是 λ=300（78.00±0.54），λ=1000 只有 77.65。
#     INR 侧旧写法取 `default` = λ=100 = **该基准上最弱的一档**（63.52，比 λ=300/1000
#     低 2.8 点）。⇒ 旧写法让两个基准用了不同的 EWC 行、且 INR 用的是最弱档，等于
#     把「FOLoRA 优于 EWC」建立在欠调基线上（FOLoRA决策日志.md §24.2 的坑）。
#     §13.3 原文：「必须始终报完整 λ 曲线」「主配置 k=64/λ=3 预先选定」「不得因为
#     几个 seed 好看就换 λ」。所以：**全列 + 判据固定 λ=100**，让读者自己看见翻转。
#     注意 INR **没有 λ=3000**（该档未在 INR 上补 n，不存在的行不能编）。
#
# (2) CODA-Prompt 改指**忠实几何** `pool100_len8_ep20`（INR 是 `..._ep20_inr`）。
#     旧写法取 `pilot20`，其真实几何是 pool=20/len=5，即**换了 L2P 的 prompt 几何**
#     ——正文声明「prompt 方法按各自原论文设置训 20 epoch」，而 pilot20 只有 epoch
#     数对、几何不对。更糟的是它在 CIFAR 是三个 CODA 配置里读数最高的那个
#     （75.83 > default 5ep 75.70 > 忠实 74.71），取它 = 挑配置。
#     L2P 的 pilot20 是干净的（pool/len 与 L2P 原论文一致，只改 epoch），保留。
#
# (3) tag 名两基准不同（CODA 忠实档 CIFAR=`pool100_len8_ep20`、INR=`pool100_len8_ep20_inr`），
#     这是历史命名，不能统一改——改了会指向不存在的目录，find_key 返回 None 后
#     只打印一行「[警告] 缺少基线」就**静默跳过该行**，表会少一行而不报错。
# ---------------------------------------------------------------------------
# P0-1 重训开关（2026-10-03 起）
# ---------------------------------------------------------------------------
# O-LoRA 与 InfLoRA 原先只把**末个任务**的 adapter 落盘，导致「评估的模型 ≠ 训练的
# 模型」；重训以 `<原 tag>_fix1` 落盘（见 scripts/retrain_fix1.py::fix1_tag）。
# 重训完成后，把下面这一处置为 "_fix1" 即可整体切换。
#
# **切换点必须只有一处**：散成多处手改必然漏掉一两处，而漏掉的表现是「表里混着新旧
# 两批 run」——不会报错、数字也都在合理范围内，是最难被发现的一类错误。
# 已有的另一处是 scripts/make_paper_tables.py 的 _appendix_exploratory()（O-LoRA 两个
# 正交强度那两行，引用同一个 `S.FIX1`，同样不要就地写死）。
#
# 切换前置条件：先跑 `python -m scripts.eval_ncm_sweep --benchmark all` 让 fix1 的
# NCM 缓存落地，否则 find_key 找不到 tag，主表会直接缺行。
# 旧 run 目录与旧缓存**保留作证**（数据完整性铁律），不得删除、不得覆盖。
# ★ 2026-10-05 09:2x 切换：35/35 个 fix1 run 已落盘（10-03 22:20 → 10-05 09:08，34.3 GPU·h），
#   抽检 + 全量核对 checkpoint 的下标集合均为 (0,19,20)（旧 run 是 (19,19,1)：olora 24 键、
#   inflora 12 键，正是 P0-1 的症状），故此处由 "" 改为 "_fix1"。
#   **前置条件**：必须先重建 fix1 的 NCM 缓存（`python -m scripts.eval_ncm_sweep
#   --benchmark cifar100` 与 `--benchmark imagenetr`，注意没有 `all` 这个取值），
#   否则下面的 tag 在 reports/ncm/ 里查不到，主表会**直接缺行**。
#   旧 run 目录、旧 NCM 缓存、旧的派生聚合（已移入 reports/_archive_pre_fix1/）
#   全部保留作证。
FIX1 = "_fix1"

PAPER_MAIN_NCM = {
    "cifar100": [
        ("Seq-LoRA", "seq", "default"),
        # 判据基线（FOLoRA决策日志.md §13.3 预注册）
        ("EWC-LoRA (lam=100)", "ewc", "default"),
        ("EWC-LoRA (lam=300)", "ewc", "ewc_lam300"),
        ("EWC-LoRA (lam=1000)", "ewc", "ewc_lam1000"),
        ("EWC-LoRA (lam=3000)", "ewc", "ewc_lam3000"),
        ("O-LoRA", "olora", f"default{FIX1}"),
        ("L2P", "l2p", "pilot20"),
        ("CODA-Prompt", "coda", "pool100_len8_ep20"),
        ("InfLoRA", "inflora", f"default{FIX1}"),
    ],
    "imagenetr": [
        ("Seq-LoRA", "seq", "default"),
        ("EWC-LoRA (lam=100)", "ewc", "default"),
        ("EWC-LoRA (lam=300)", "ewc", "ewc_lam300"),
        ("EWC-LoRA (lam=1000)", "ewc", "ewc_lam1000"),
        # INR 无 λ=3000
        ("O-LoRA", "olora", f"default{FIX1}"),
        ("L2P", "l2p", "pilot20"),
        ("CODA-Prompt", "coda", "pool100_len8_ep20_inr"),
        ("InfLoRA", "inflora", f"default{FIX1}"),
    ],
}

# 本文方法。results 口径下是原默认配置 λ=300 / k=16。
OURS = ("FOLoRA", "folora_v2", "v2f_l300_k16")
# NCM 口径下 = **§13.3 预注册的主配置**（k=64, λ=3），不是「跑完之后挑最优点」。
# 旧值写的是 v2f_l10_k64（λ=10）——那是在 λ 网格上事后挑一个读数高的点，正是 §13.3
# 明文禁止的「挑 λ」。λ=10 确实略高（78.24 n=5 vs 77.87 n=10），但它不是预注册配置，
# 拿它当本文方法会让全表结论偏向自己。λ=10 作为网格点照常在消融表里报。
OURS_NCM = ("FOLoRA", "folora_v2", "v2f_l3_k64")

# 论文正文引用的 p 值（05_experiments.tex），供 --check-paper 回验。
# 正文定的口径是 paired（各方法 seed 已对齐），只对「FOLoRA vs EWC-LoRA」给了具体数值。
PAPER_TEST = "paired"
# !! 2026-10-03 已回填，值来自本脚本在 NCM 口径下的实测（主配置 v2f_l3_k64，
# 判据基线 EWC λ=100，paired，common seeds）。**这些就是正文要印的数字**；
# 正文与这里必须逐位一致，改了正文就要同步改这里——两边漂移正是本表存在的原因。
# 先前一律置 None 是因为旧的线性头 CIL 数字在 NCM 下全部作废，不能拿去比。
# 注意：这里只放「FOLoRA vs EWC λ=100」这一对（§13.3 的判据基线）。
# λ=300/1000/3000 上的结果（含 FGT 方向显著反向）写在正文与 λ 敏感性表里，
# 不塞进这个回验表——回验只钉住判据那一对，避免把「判决」和「敏感性」混为一谈。
PAPER_CLAIMED = {
    ("cifar100", "forgetting_cil"): 0.0707,
    ("cifar100", "final_acc_cil"): 0.0734,
    ("imagenetr", "forgetting_cil"): 0.0004,
    ("imagenetr", "final_acc_cil"): 0.0002,
}

METRICS = [("final_acc_cil", "avgACC(%)"), ("forgetting_cil", "FGT(%)")]


def load_runs(root: Path, benchmark: str):
    """返回 {(method, tag, proto): {seed: results}}，与 aggregate.py 同逻辑。"""
    grouped = defaultdict(dict)
    for rpath in root.glob(f"{benchmark}/**/seed*/results.json"):
        d = json.loads(rpath.read_text(encoding="utf-8"))
        cfg = d["config"]
        proto = (cfg.get("num_tasks"), cfg.get("epochs"), cfg.get("lora_rank"))
        grouped[(cfg["method"], cfg.get("tag", ""), proto)][cfg["seed"]] = d
    return grouped


def _proto_of_run(run_dir):
    """回读 <run_dir>/config.json 取协议三元组（NCM 缓存本身不存 epochs/lora_rank）。"""
    if not run_dir:
        return None
    d = read_json_or_none(Path(run_dir) / "config.json")
    if not d:
        return None
    return (d.get("num_tasks"), d.get("epochs"), d.get("lora_rank"))


def load_runs_ncm(cache_root: Path, benchmark: str):
    """返回 {(method, tag, proto): {seed: rec}}，读 NCM 评估缓存（论文现用协议）。

    与 load_runs 同结构，所以 compare/values/find_key 都能直接复用。
    协议三元组不在缓存里，得回每条记录各自的 config.json 取——**不能只看 num_tasks**：
    `*/pilot20` 那批 num_tasks 同样是 20 但 epochs=20，精度低 5~10 点，混进来结论全错。
    过滤按 `expected_proto(method)` 逐方法判定（prompt 类本就是 20 epoch）。
    """
    grouped = defaultdict(dict)
    base = cache_root / benchmark
    if not base.exists():
        return grouped
    for cpath in sorted(base.glob("*.json")):
        rec = read_json_or_none(cpath)
        if not rec or rec.get("method") == "simplecil":
            continue                    # 冻结特征基线 n=1，无方差，不参与检验
        method = rec["method"]
        p = _proto_of_run(rec.get("run_dir"))
        if p is None or p != expected_proto(method):
            continue
        grouped[(method, rec["tag"], p)][rec["seed"]] = rec
    return grouped


def find_key(grouped, method, tag):
    """按 (method, tag) 找唯一匹配的 protocol；多个则取 seed 数最多的那个。"""
    hits = [(k, v) for k, v in grouped.items() if k[0] == method and k[1] == tag]
    if not hits:
        return None
    return max(hits, key=lambda kv: len(kv[1]))[0]


def values(seeds: dict, metric: str):
    """按 seed 排序取值（×100 换成百分数）。"""
    return [seeds[s][metric] * 100 for s in sorted(seeds)]


def run_test(a, b, paired: bool):
    """返回 (t, p, n_a, n_b, n_used)。paired 时要求 a、b **已按 seed 值对齐且等长**
    （对齐由 compare 完成，见那里的注释）。

    n<2 时**提前返回 nan**，不要交给 scipy：它会在内部算样本方差时报
    "divide by zero" 警告刷屏，而调用方本来就会把这种情况标成 n/a。
    """
    n_used = min(len(a), len(b)) if paired else len(a)
    if n_used < 2 or (not paired and len(b) < 2):
        return float("nan"), float("nan"), len(a), len(b), n_used
    if paired:
        res = stats.ttest_rel(a[:n_used], b[:n_used])
    else:
        res = stats.ttest_ind(a, b, equal_var=False)
    return res.statistic, res.pvalue, len(a), len(b), n_used


FAMILY_MODES = ("bench-metric", "bench", "global", "none")
FAMILY_HELP = {
    "bench-metric": "同一 benchmark、同一指标下的全部对照（默认；主表 m≈6）",
    "bench": "同一 benchmark 的两个指标并成一个家族（m 翻倍，更保守）",
    "global": "跨全部 benchmark 与指标一个家族（最保守，m≈24）",
    "none": "不校正（只留原始 p；仅供与旧产物逐位对照）",
}


def apply_holm(rows, mode="bench-metric", alpha=0.05):
    """Holm–Bonferroni step-down 校正，**就地**写入每行的
    `p_holm` / `significant_holm` / `family` / `family_size` / `alpha`。

    家族（family）怎么划是结论的一部分，不是实现细节——同一批 p 值换个家族划分
    就可能从「全部显著」变成「一半不显著」——所以家族名与大小都落盘，正文照抄。

    不可检验的条目（单 seed / 零方差 → p 为 nan）**不进入家族**，m 只数有效 p。
    「这个检验没做」与「做了且不显著」是两回事：把 nan 当 p=1 计入会把 m 撑大、
    阈值变松，等于凭空放宽其余条目，方向恰好是反的。

    返回 {家族名: 家族大小}，供调用方如实记录家族的构成。
    """
    # 先给所有行补上字段，保证 --family none 时产物 schema 仍与校正时一致
    # （下游读 p_holm 的代码不必区分两种模式）。
    for r in rows:
        r.setdefault("p_holm", None)
        r.setdefault("significant_holm", None)
        r.setdefault("family", None)
        r.setdefault("family_size", None)
        r.setdefault("alpha", alpha)
    if mode == "none":
        return {}

    def key_of(r):
        if mode == "global":
            return ("all",)
        if mode == "bench":
            return (r["benchmark"],)
        return (r["benchmark"], r["metric"])

    fams = defaultdict(list)
    for r in rows:
        fams[key_of(r)].append(r)

    sizes = {}
    for fam_key, fam in sorted(fams.items(), key=lambda kv: str(kv[0])):
        name = "/".join(fam_key)
        # p 可能是 None（未检验）或 nan（scipy 对零方差/单样本返回 nan）：两者都不可检验。
        # `p != p` 是 nan 判定，不能写成 `p is None`——nan 不是 None，会漏掉。
        valid = [r for r in fam if r["p"] is not None and r["p"] == r["p"]]
        m = len(valid)
        sizes[name] = m
        for r in fam:
            r.update({"family": name, "family_size": m})
        if m == 0:
            for r in fam:
                r.update({"p_holm": None, "significant_holm": False})
            continue
        order = sorted(range(m), key=lambda i: valid[i]["p"])
        running = 0.0
        for rank, i in enumerate(order):
            # Holm 的校正 p：p_(k)·(m−k+1)，再沿排序单调化（取前缀最大值）。
            # 单调化之后「p_holm ≤ α」与教科书的 step-down 判定逐条等价，
            # 但省掉了「第一个不满足就全停」的状态机，也不会漏标。
            running = max(running, min(1.0, valid[i]["p"] * (m - rank)))
            valid[i].update({"p_holm": float(running),
                             "significant_holm": bool(running <= alpha)})
        valid_ids = {id(r) for r in valid}
        for r in fam:
            if id(r) not in valid_ids:
                r.update({"p_holm": None, "significant_holm": False})
    return sizes


def compare(grouped, benchmark: str, ours_key, base_key, metric: str, paired: bool):
    ours_seeds = grouped[ours_key]
    base_seeds = grouped[base_key]
    if paired:
        # 按**实际 seed 值**取交集来配对。绝不能各自按 seed 排序后截断对齐：
        # 两边 seed 集合经常不同（如 O-LoRA 缺 seed2/3，只有 [0,1,4..9]），
        # 截断会把 (ours seed2 ↔ base seed4) 这种不相干的两条配成一对，
        # 算出来的 p 值是假的、且**不会报错**——正是最危险的那类 bug。
        common = sorted(set(ours_seeds) & set(base_seeds))
        a = [ours_seeds[s][metric] * 100 for s in common]
        b = [base_seeds[s][metric] * 100 for s in common]
    else:
        common = None
        a, b = values(ours_seeds, metric), values(base_seeds, metric)
    t, p, n_a, n_b, n_used = run_test(a, b, paired)
    # 两侧 seed 交集可能为空（如两方法没有任何共同 seed）。run_test 会返回 nan，
    # 但下面这行除法不会：`sum([]) / 0` 抛 ZeroDivisionError，把整轮显著性检验打断，
    # 而不是产出一行 n/a。空列表给 nan 与 run_test 的口径一致。
    m_a = sum(a) / len(a) if a else float("nan")
    m_b = sum(b) / len(b) if b else float("nan")

    # 记录**双侧**的 key。此前只写 base 侧，导致 --all-configs 的产物里
    # `ours_mean` 被钉死在主配置（加权 k=64, λ=3）而读者无从得知，
    # 把「等权 λ=10」那行误读成「同 λ 下的加权 vs 等权」——论文 §5.3 的
    # λ=10 数字（+0.46, p=0.626）就是这么做出来的，真值是 −0.03, p=0.985。
    # **ours 侧必须显式落盘，否则同名 artifact 迟早被再次误读。**
    ours_meta, base_meta = {}, {}
    if isinstance(ours_key, tuple):
        ours_meta = {"ours_method": ours_key[0], "ours_tag": ours_key[1],
                     "ours_proto": list(ours_key[2]) if len(ours_key) > 2 else None}
    if isinstance(base_key, tuple):
        # base 侧原先只落 method + proto，**漏了 tag**，而 ours 侧落了 ours_tag。
        # 这种不对称正是上面那条误读的成因之一：读者能确认 ours 臂是哪个配置，
        # 却无从确认 base 臂是哪个（EWC 有 λ=30/100/300/1000/3000 五个 tag，
        # 取错一个表行数字全变）。两侧字段必须对称。
        base_meta = {"base_method": base_key[0],
                     "base_tag": base_key[1] if len(base_key) > 1 else None,
                     "base_proto": list(base_key[2]) if len(base_key) > 2 else None}

    # P3-9：论文有三处「每个 seed 的差值都同号」这类断言
    # （05_experiments.tex:359 "all ten per-seed differences positive"、:390
    #  "every per-seed difference of the same"、:577 "seed-wise all of one sign"）。
    # 此前产物只存**聚合均值 + p 值**，这三句无法从 artifact 独立复核 —— 只能重跑。
    # 逐 seed 差值落盘后，「同号」直接数得出来。单位是百分点（与 delta 一致）。
    per_seed = {}
    if paired and common:
        per_seed = {
            str(s): float(ours_seeds[s][metric] - base_seeds[s][metric]) * 100
            for s in common
        }
    n_pos = sum(1 for v in per_seed.values() if v > 0)
    n_neg = sum(1 for v in per_seed.values() if v < 0)
    n_zero = sum(1 for v in per_seed.values() if v == 0)

    # 全部转成 Python 原生类型，否则 numpy.float64 / numpy.bool_ 无法 json 序列化
    return {
        "benchmark": benchmark, "metric": metric,
        "ours_mean": float(m_a), "base_mean": float(m_b), "delta": float(m_a - m_b),
        "n_ours": int(n_a), "n_base": int(n_b), "n_used": int(n_used),
        "paired_seeds": common,
        # 逐 seed 差值 + 符号统计（见上方 P3-9 注释）。n_used<2 时 per_seed 为空，
        # 此时 same_sign 为 False（"无证据" ≠ "同号"）。
        "per_seed_deltas": per_seed,
        "n_positive": int(n_pos), "n_negative": int(n_neg), "n_zero": int(n_zero),
        "same_sign": bool(n_pos + n_neg > 0 and (n_pos == 0 or n_neg == 0)),
        "t": float(t), "p": float(p), "significant": bool(p < 0.05),
        **ours_meta, **base_meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="cifar100",
                    choices=["cifar100", "imagenetr", "all"])
    ap.add_argument("--source", default="ncm", choices=["ncm", "results"],
                    help="ncm=冻结特征+NCM 协议（论文现用，默认）；"
                         "results=旧的「随任务增长的线性头」CIL 口径，仅供对照")
    ap.add_argument("--out_dir", default="experiments")
    ap.add_argument("--ncm_root", default=str(NCM_CACHE_ROOT))
    ap.add_argument("--ours-tag", default=None,
                    help="覆盖本文方法的 tag；NCM 口径默认 v2f_l3_k64（§13.3 预注册主配置）")
    ap.add_argument("--test", default="paired", choices=["welch", "paired"],
                    help="paired=配对 t 检验（默认，论文口径，§13.3 预注册）；"
                         "welch=独立样本 Welch t 检验，仅供对照")
    ap.add_argument("--all-configs", action="store_true",
                    help="扫全部 config（含消融与 EWC 调优），而非只跑论文主表")
    ap.add_argument("--family", default="bench-metric", choices=list(FAMILY_MODES),
                    help="Holm 校正的家族划分；"
                         + "；".join(f"{k}={v}" for k, v in FAMILY_HELP.items())
                         + "。换家族必须在论文正文同步写明（家族是结论的一部分）")
    ap.add_argument("--check-paper", action="store_true",
                    help="把实测 p 值与论文正文引用的数值对照")
    args = ap.parse_args()

    paired = args.test == "paired"
    benchmarks = ["cifar100", "imagenetr"] if args.benchmark == "all" else [args.benchmark]

    use_ncm = args.source == "ncm"
    paper_main = PAPER_MAIN_NCM if use_ncm else PAPER_MAIN
    ours = ((OURS_NCM[0], OURS_NCM[1], args.ours_tag or OURS_NCM[2])
            if use_ncm else OURS)
    print(f"数据来源 = {args.source}"
          f"{'（冻结特征 + NCM，论文协议）' if use_ncm else '（旧口径：线性头 CIL）'}"
          f"；本文方法 = {ours[1]}/{ours[2]}")

    all_rows, all_print = [], []
    loaded = {}          # bench -> (grouped, ours_key)，供 --check-paper 复用
    rows_by_bench = {}   # bench -> [row]。**先算全部、再校正、最后才打印**：
    #                      Holm 必须先看到同一家族的所有 p 才知道 m，边算边印拿不到。
    for bench in benchmarks:
        grouped = (load_runs_ncm(Path(args.ncm_root), bench) if use_ncm
                   else load_runs(Path(args.out_dir), bench))
        ours_key = find_key(grouped, ours[1], ours[2])
        if ours_key is None:
            print(f"[跳过] {bench}：找不到 {ours[1]}/{ours[2]} 的结果")
            continue
        loaded[bench] = (grouped, ours_key)

        if args.all_configs:
            # 同 protocol 的全部 config 互比（避免跨协议混算）
            targets = [k for k in sorted(grouped)
                       if k[2] == ours_key[2] and k != ours_key]
        else:
            targets = []
            for label, method, tag in paper_main[bench]:
                k = find_key(grouped, method, tag)
                if k is None:
                    print(f"[警告] {bench}：缺少基线 {method}/{tag}，跳过")
                    continue
                targets.append(k)

        rows = []
        for k in targets:
            label = next((l for l, m, t in paper_main[bench] if (m, t) == (k[0], k[1])),
                         f"{k[0]}/{k[1] or '-'}")
            proto = f"{k[2][0]}t/{k[2][1]}e/r{k[2][2]}"
            for metric, mname in METRICS:
                r = compare(grouped, bench, ours_key, k, metric, paired)
                r.update({"baseline": label, "baseline_tag": k[1], "proto": proto,
                          "test": args.test, "method": k[0], "metric_name": mname})
                # 单 seed / 零方差时 t 检验无定义（scipy 会返回 nan），如实标注而不是打印 nan。
                # 必须在 apply_holm 之前标记：nan 不是 None，apply_holm 靠 `p != p` 判它，
                # 但这里把 p 改写成 None 之后语义一致（None 与 nan 都不可检验），
                # 且下游 JSON 里 `"p": null` 比 `NaN` 更易读（NaN 不是合法 JSON）。
                if min(r["n_ours"], r["n_base"]) < 2 or r["p"] != r["p"]:
                    r.update({"t": None, "p": None, "significant": False,
                              "note": "单 seed 或零方差，t 检验无定义"})
                rows.append(r)
        rows_by_bench[bench] = rows
        all_rows.extend(rows)

    # Holm–Bonferroni：家族由 --family 决定，家族构成落盘（family/family_size）
    fam_sizes = apply_holm(all_rows, args.family)
    if args.family != "none":
        fam_txt = "；".join(f"{k}(m={v})" for k, v in sorted(fam_sizes.items()))
        print(f"Holm 校正：家族={args.family}（{FAMILY_HELP[args.family]}）—— {fam_txt}")

    for bench in benchmarks:
        rows = rows_by_bench.get(bench)
        if not rows:
            continue
        header = (f"\n=== {bench}：{OURS[0]} vs 基线"
                  f"（{'配对' if paired else 'Welch'} t 检验；"
                  f"* = Holm 校正后 p<0.05，家族={args.family}）===")
        table = [header,
                 f"{'baseline':<22}{'proto':<12}{'n':>5} {'metric':<16}"
                 f"{'ours':>9}{'base':>9}{'diff':>9}{'t':>9}{'p':>9}{'p_holm':>9} sig"]
        for r in rows:
            head = (f"{r['baseline']:<22}{r['proto']:<12}{r['n_used']:>5}"
                    f" {r['metric_name']:<16}"
                    f"{r['ours_mean']:>9.2f}{r['base_mean']:>9.2f}{r['delta']:>+9.2f}")
            # `is None` 是**正确**判定，不是漏判 nan：上面构造 rows 时（见本函数开头
            # 「单 seed / 零方差」那段）已把 nan 统一改写成 None。补 `p != p` 只为防
            # 未来有人绕过那段归一化直接 append 行 —— 属加固，不是修 bug。
            if r["p"] is None or r["p"] != r["p"]:
                table.append(f"{head}{'n/a':>9}{'n/a':>9}{'n/a':>9}   -")
                continue
            ph = r["p_holm"]
            star = "*" if r["significant_holm"] else " "
            # 遗忘越低越好，diff 取「ours-base」；准确率越高越好
            table.append(f"{head}{r['t']:>+9.3f}{r['p']:>9.4f}"
                         f"{(f'{ph:.4f}' if ph is not None else 'n/a'):>9}{star}")
        block = "\n".join(table)
        print(block)
        all_print.append(block)

    if args.check_paper:
        # 论文口径是 paired，所以这里固定按 paired 复算，与 --test 无关
        note = "" if args.test == PAPER_TEST else f"（论文口径 = {PAPER_TEST}，已按其复算）"
        print(f"\n=== 回验：论文正文引用的 p 值 vs 本脚本实测 {note}===")
        print(f"{'benchmark':<12}{'metric':<16}{'论文':>10}{'实测':>10}{'':>6}")
        for (bench, metric), claimed in PAPER_CLAIMED.items():
            if bench not in loaded:
                continue
            grouped, ours_key = loaded[bench]
            # EWC 对照取主表里的**第一个** ewc 行，即判据基线。
            # NCM 口径下 PAPER_MAIN_NCM 把 λ=100（tag `default`）列在首位，正是 §13.3
            # 预注册的判据基线；results 口径是 default→config tag 为空串。
            # **顺序有语义**：调整 PAPER_MAIN_NCM 里 EWC 行的先后会改变本回验的对照基准。
            ewc_tag = next((t for l, m, t in paper_main[bench] if m == "ewc"), "")
            base_key = find_key(grouped, "ewc", ewc_tag)
            if base_key is None:
                continue
            if claimed is None:
                r = compare(grouped, bench, ours_key, base_key, metric, paired=True)
                # 同上：r 未归一化，nan 时不能印成回填值（"nan" 不是可回填的 p 值）
                if r["p"] is None or r["p"] != r["p"]:
                    print(f"{bench:<12}{metric:<16}{'(待回填)':>10}{'n/a':>10}"
                          f"  不可检验，无从回填")
                    continue
                print(f"{bench:<12}{metric:<16}{'(待回填)':>10}{r['p']:>10.4f}"
                      f"  ← 用这个值回填 PAPER_CLAIMED")
                continue
            r = compare(grouped, bench, ours_key, base_key, metric, paired=True)
            # 本路径的 r **未经**上面的 nan→None 归一化（直接取 compare 的返回值），
            # 所以这里必须同时挡裸 nan，否则 `abs(nan - claimed) < 5e-4` 恒为 False，
            # 会把它误判成「不符」。
            if r["p"] is None or r["p"] != r["p"]:
                print(f"{bench:<12}{metric:<16}{'n/a':>10}{'n/a':>10}  无法检验")
                continue
            # claimed 必非 None：上面的 `if claimed is None: ... continue` 已把 None
            # 分支提前处理并跳过，此处原先残留的 `if claimed is None` 是**不可达死代码**，
            # 已删（P3-13）；保留的这行才是实际判定。
            verdict = "OK" if abs(r["p"] - claimed) < 5e-4 else "不符"
            shown = f"{claimed:.3f}"
            print(f"{bench:<12}{metric:<16}{shown:>10}{r['p']:>10.4f}{verdict:>8}")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    suffix = f"{args.benchmark}{'_paired' if paired else ''}" \
             f"{'_all' if args.all_configs else ''}{'' if use_ncm else '_headcil'}" \
             f"{'_holm-' + args.family if args.family != 'none' else '_noholm'}"
    (out / f"significance_{suffix}.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# 显著性检验（{args.test} t-test，Holm 家族={args.family}）\n",
          "对照 = 论文 Table 1 的主表。FGT 越小越好（diff 为负表示本文方法遗忘更低），"
          "avgACC 越大越好。\n",
          f"- `p` = 原始配对 p 值（未校正）",
          f"- `p_holm` = Holm–Bonferroni 校正后 p；`*` 标在 `p_holm<0.05` 上，"
          f"即**本文对外声明的显著性口径**",
          f"- 家族划分 = `{args.family}`：{FAMILY_HELP[args.family]}"
          + (f"；本次家族大小 {fam_sizes}" if fam_sizes else ""),
          f"- 不可检验条目（单 seed / 零方差）不计入家族："
          f"「没做这个检验」与「做了且不显著」不能混算\n",
          "```", *all_print, "```"]
    (out / f"significance_{suffix}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n已写入 reports/significance_{suffix}.md 和 .json")


if __name__ == "__main__":
    main()
