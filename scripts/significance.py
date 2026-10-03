"""显著性检验：论文表格里的「FOLoRA 优于基线」是否有统计支撑。

用法：
    python -m scripts.significance                    # 论文主表（NCM 协议 + paired t，默认）
    python -m scripts.significance --benchmark all    # 两个基准一起
    python -m scripts.significance --all-configs      # 扫全部 config（含消融/调优）
    python -m scripts.significance --test welch       # 退回 Welch 独立样本 t，仅供对照
    python -m scripts.significance --check-paper      # 回验论文正文引用的 p 值
    python -m scripts.significance --source results   # 退回旧口径（线性头 CIL），仅供对照

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
PAPER_MAIN_NCM = {
    "cifar100": [
        ("Seq-LoRA", "seq", "default"),
        # 判据基线（FOLoRA决策日志.md §13.3 预注册）
        ("EWC-LoRA (lam=100)", "ewc", "default"),
        ("EWC-LoRA (lam=300)", "ewc", "ewc_lam300"),
        ("EWC-LoRA (lam=1000)", "ewc", "ewc_lam1000"),
        ("EWC-LoRA (lam=3000)", "ewc", "ewc_lam3000"),
        ("O-LoRA", "olora", "default"),
        ("L2P", "l2p", "pilot20"),
        ("CODA-Prompt", "coda", "pool100_len8_ep20"),
        ("InfLoRA", "inflora", "default"),
    ],
    "imagenetr": [
        ("Seq-LoRA", "seq", "default"),
        ("EWC-LoRA (lam=100)", "ewc", "default"),
        ("EWC-LoRA (lam=300)", "ewc", "ewc_lam300"),
        ("EWC-LoRA (lam=1000)", "ewc", "ewc_lam1000"),
        # INR 无 λ=3000
        ("O-LoRA", "olora", "default"),
        ("L2P", "l2p", "pilot20"),
        ("CODA-Prompt", "coda", "pool100_len8_ep20_inr"),
        ("InfLoRA", "inflora", "default"),
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
    m_a, m_b = sum(a) / len(a), sum(b) / len(b)
    # 全部转成 Python 原生类型，否则 numpy.float64 / numpy.bool_ 无法 json 序列化
    return {
        "benchmark": benchmark, "metric": metric,
        "ours_mean": float(m_a), "base_mean": float(m_b), "delta": float(m_a - m_b),
        "n_ours": int(n_a), "n_base": int(n_b), "n_used": int(n_used),
        "paired_seeds": common,
        "t": float(t), "p": float(p), "significant": bool(p < 0.05),
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

        header = (f"\n=== {bench}：{OURS[0]} vs 基线"
                  f"（{'配对' if paired else 'Welch'} t 检验，* = p<0.05）===")
        table = [header,
                 f"{'baseline':<22}{'proto':<12}{'n':>5} {'metric':<16}"
                 f"{'ours':>9}{'base':>9}{'diff':>9}{'t':>9}{'p':>10}"]
        for k in targets:
            label = next((l for l, m, t in paper_main[bench] if (m, t) == (k[0], k[1])),
                         f"{k[0]}/{k[1] or '-'}")
            proto = f"{k[2][0]}t/{k[2][1]}e/r{k[2][2]}"
            for metric, mname in METRICS:
                r = compare(grouped, bench, ours_key, k, metric, paired)
                r.update({"baseline": label, "baseline_tag": k[1], "proto": proto,
                          "test": args.test, "method": k[0]})
                all_rows.append(r)
                head = (f"{label:<22}{proto:<12}{r['n_used']:>5} {mname:<16}"
                        f"{r['ours_mean']:>9.2f}{r['base_mean']:>9.2f}{r['delta']:>+9.2f}")
                # 单 seed / 零方差时 t 检验无定义（scipy 会返回 nan），如实标注而不是打印 nan
                if min(r["n_ours"], r["n_base"]) < 2 or r["p"] != r["p"]:
                    r.update({"t": None, "p": None, "significant": False,
                              "note": "单 seed 或零方差，t 检验无定义"})
                    table.append(f"{head}{'n/a':>9}{'n/a':>10} ")
                    continue
                star = "*" if r["significant"] else " "
                # 遗忘越低越好，diff 取「ours-base」；准确率越高越好
                table.append(f"{head}{r['t']:>+9.3f}{r['p']:>9.4f}{star}")

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
                print(f"{bench:<12}{metric:<16}{'(待回填)':>10}{r['p']:>10.4f}"
                      f"  ← 用这个值回填 PAPER_CLAIMED")
                continue
            r = compare(grouped, bench, ours_key, base_key, metric, paired=True)
            if r["p"] is None:
                print(f"{bench:<12}{metric:<16}{'n/a':>10}{'n/a':>10}  无法检验")
                continue
            if claimed is None:
                verdict, shown = ("OK" if r["p"] < 0.001 else "不符"), "<0.001"
            else:
                verdict = "OK" if abs(r["p"] - claimed) < 5e-4 else "不符"
                shown = f"{claimed:.3f}"
            print(f"{bench:<12}{metric:<16}{shown:>10}{r['p']:>10.4f}{verdict:>8}")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    suffix = f"{args.benchmark}{'_paired' if paired else ''}" \
             f"{'_all' if args.all_configs else ''}{'' if use_ncm else '_headcil'}"
    (out / f"significance_{suffix}.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# 显著性检验（{args.test} t-test）\n",
          "对照 = 论文 Table 1 的主表；`*` 表示 p<0.05。"
          "FGT 越小越好（diff 为负表示本文方法遗忘更低），avgACC 越大越好。\n",
          "```", *all_print, "```"]
    (out / f"significance_{suffix}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n已写入 reports/significance_{suffix}.md 和 .json")


if __name__ == "__main__":
    main()
