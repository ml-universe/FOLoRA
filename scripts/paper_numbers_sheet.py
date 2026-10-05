"""把论文散文里**手写的数字**一次性读出来，供重训/重评估后逐条对账。

为什么需要它
------------
表体由 `make_paper_tables.py` 生成，所以表里的数字不会漂；但正文里有几段是**手写**
的具体值（`+9.93`、`p=0.012`、`68.44\\pm0.50`、`+3.61` 到 `+13.07`、距地板多少……）。
这些数字一旦随重训变化，没有任何机制会提醒作者——编译通过、表也自洽，只有正文悄悄
过时。本脚本把每个这样的值都从**数据**算一遍打印出来，作为改正文时的唯一来源。

**不与表格生成器重复实现统计**：均值/std 取自 `make_paper_tables.load_summary`
（即各表自己的口径），Δ 与 p 取自 `make_paper_tables.confirmatory_holm`（即各表星号
自己的口径）。这里只做「读出来 + 排版」，**不重算**——另写一份统计实现迟早会与表体
给出不同的显著性判定，而两者在论文里并排印着。

用法
----
    python -m scripts.paper_numbers_sheet
    python -m scripts.paper_numbers_sheet --benchmark cifar100
"""

import argparse
import json
from pathlib import Path

from scripts import significance as S
from scripts.make_paper_tables import (
    MAIN_ROWS, OURS, confirmatory_holm, load_summary,
)
from scripts.significance import compare, find_key, load_runs_ncm

# 正文 §5.2 逐条点名的四个 PEFT 基线（与 05_experiments.tex 的句子一一对应）。
PEFT_FAMILY = {
    "cifar100": [("L2P", "l2p", "pilot20"),
                 ("CODA-Prompt", "coda", "pool100_len8_ep20"),
                 ("InfLoRA", "inflora", f"default{S.FIX1}"),
                 ("O-LoRA", "olora", f"default{S.FIX1}")],
    "imagenetr": [("L2P", "l2p", "pilot20"),
                  ("CODA-Prompt", "coda", "pool100_len8_ep20_inr"),
                  ("InfLoRA", "inflora", f"default{S.FIX1}"),
                  ("O-LoRA", "olora", f"default{S.FIX1}")],
}

# §5.4 引用的是 O-LoRA **训练期**正交约束的两个强度（与本文的 Fisher 加权对照）。
OLORA_ORTH = [("0.1", f"olora_orth_l0.1{S.FIX1}"), ("1", f"olora_orth_l1{S.FIX1}")]

FLOOR = ("simplecil", "frozen")

# `reports/ncm_summary_<bench>.json` 的度量键与 NCM 缓存（`reports/ncm/*.json`）**不同名**：
# 汇总层是短名 `acc` / `fgt`（值为**分数**，0.7031），缓存层是 `final_acc_cil` /
# `forgetting_cil`（值同为分数）。此处显式映射并注释，因为写错的表现是**全部行都为 None**
# ——不会报错，只会打印一片「缺」，极易被当成「数据还没生成」而放过（本文件初版即如此）。
_SUMMARY_KEY = {"final_acc_cil": "acc", "forgetting_cil": "fgt"}


def _cell(summ, method, tag, key="final_acc_cil"):
    """返回 (mean, std, n)，**单位为百分数**；缺行返回 None。"""
    rec = summ.get(f"{method}/{tag}")
    if rec is None:
        return None
    m = rec.get(f"{_SUMMARY_KEY[key]}_mean")
    if m is None:
        return None
    return (m * 100.0, (rec.get(f"{_SUMMARY_KEY[key]}_std") or 0.0) * 100.0,
            rec.get("n"))


def _bench(bench: str, bench_tag: str):
    try:
        summ = load_summary(bench)
    except SystemExit as e:
        print(f"\n[{bench}] 读不到汇总：{e}")
        return
    try:
        fam = confirmatory_holm(bench)
    except SystemExit as e:
        print(f"\n[{bench}] 确认性家族算不出：{e}")
        return
    print(f"\n{'='*80}\n== {bench_tag}   (FIX1={S.FIX1!r}，n 来自汇总)\n{'='*80}")

    floor = _cell(summ, *FLOOR)
    ours = _cell(summ, *OURS)
    print("\n-- 地板与主配置 --")
    print(f"   floor  SimpleCIL(冻结特征) = "
          + (f"{floor[0]:.2f} (n={floor[2]})" if floor else "缺"))
    print(f"   ours   FOLoRA {OURS[1]:14s} = "
          + (f"{ours[0]:.2f} +- {ours[1]:.2f} (n={ours[2]})" if ours else "缺"))

    print("\n-- 各主表行相对地板（+ = 在地板之上；**重训可能翻转这个方向**）--")
    for label, method, tc, ti in MAIN_ROWS:
        tag = tc if bench == "cifar100" else ti
        if method == "simplecil":
            continue
        c = _cell(summ, method, tag)
        if c is None or floor is None:
            print(f"   {label:22s} 缺（{method}/{tag}）")
            continue
        d = c[0] - floor[0]
        print(f"   {label:22s} {c[0]:6.2f} +- {c[1]:5.2f} (n={c[2]:2})  "
              f"距地板 {d:+6.2f}  {'地板之上' if d > 0 else '**地板之下**'}"
              + ("   <- 本文方法" if method == OURS[0] else ""))

    print("\n-- §5.2 PEFT 家族：vs FOLoRA 的配对差（正文写的是 Δ 范围与校正 p 上限）--")
    deltas, ps, ns = [], [], set()
    for label, method, tag in PEFT_FAMILY[bench]:
        row = fam.get((method, tag, "final_acc_cil"))
        if row is None:
            print(f"   {label:14s} 不在确认性家族里（{method}/{tag}）")
            continue
        # compare() 的 delta 已经是**百分点**（内部 a/b 都乘过 100），此处不得再乘。
        deltas.append(row["delta"])
        ns.add(row.get("n_used"))
        if row.get("p_holm") is not None:
            ps.append(row["p_holm"])
        print(f"   {label:14s} n={row.get('n_used')}  Δ={row['delta']:+6.2f}  "
              f"p={row['p']:.4f}  p_Holm={row['p_holm']:.4f}"
              if row.get("p_holm") is not None else
              f"   {label:14s} n={row.get('n_used')}  Δ={row['delta']:+6.2f}  "
              f"p={row['p']:.4f}  p_Holm=n/a")
    if deltas:
        print(f"   >>> Δ 范围 [{min(deltas):+.2f}, {max(deltas):+.2f}]；n 集合 {sorted(ns)}")
    if ps:
        print(f"   >>> 家族内最大校正 p = {max(ps):.4f}"
              f"（正文写 'corrected $p\\le X$' 时取此值向上取整）")

    print("\n-- §5.4：O-LoRA 训练期正交约束两个强度（探索性，不在确认性家族内）--")
    grouped = load_runs_ncm(Path("reports/ncm"), bench)
    ours_key = find_key(grouped, *OURS)
    for name, tag in OLORA_ORTH:
        c = _cell(summ, "olora", tag)
        if c is None:
            print(f"   olora_orth_l{name:4s} 缺（{tag}）")
            continue
        line = f"   olora_orth_l{name:4s} ACC={c[0]:6.2f} +- {c[1]:5.2f} (n={c[2]})"
        bk = find_key(grouped, "olora", tag)
        if ours_key is not None and bk is not None:
            r = compare(grouped, bench, ours_key, bk, "final_acc_cil", paired=True)
            line += f"  vs FOLoRA Δ={r['delta']:+6.2f} p={r['p']:.4f} n={r['n_used']}"
        print(line)


def _curves(bench: str, bench_tag: str):
    """§5.2 曲线段（05_experiments.tex 约 406-414 行）里手写的数字。

    那一段引用的是**候选类数轴**上的首点/末点与 FOLoRA-vs-EWC 的逐点差。数字随
    (a) 矩阵的 seed 预算、(b) O-LoRA 换 tag 而变，而它在正文里是纯手写的 —— 故此节
    从 `reports/forgetting_curve_<bench>.json` 现算。
    """
    p = Path(f"reports/forgetting_curve_{bench}.json")
    if not p.exists():
        print(f"\n[{bench}] 没有 {p}（先跑 python -m scripts.plot_forgetting_curve）")
        return
    meta = json.loads(p.read_text(encoding="utf-8"))
    methods, series = meta["methods"], meta["series"]
    print(f"\n-- §5.2 曲线段（{bench_tag}）   seed 一致={meta.get('n_uniform')} --")
    print(f"   {'method':12s} {'n':>3s}  {'tag':22s} {'ACC@t1':>7s} {'ACC@t20':>8s} {'FGT@t20':>8s}")
    for m in series:                       # series 的键序 = 图例序（METHOD_ORDER）
        rec = methods.get(m, {})
        a = series[m]["avg_accuracy_seen_so_far"]
        f = series[m]["avg_forgetting_prev_tasks"]
        print(f"   {m:12s} {rec.get('n'):>3}  {str(rec.get('tag')):22s} "
              f"{a[0]:7.1f} {a[-1]:8.1f} {f[-1]:8.2f}")
    accs_first = {m: series[m]["avg_accuracy_seen_so_far"][0] for m in series}
    accs_last = {m: series[m]["avg_accuracy_seen_so_far"][-1] for m in series}
    print(f"   >>> t=1 六曲线跨度 {min(accs_first.values()):.1f} 至 "
          f"{max(accs_first.values()):.1f}（差 {max(accs_first.values())-min(accs_first.values()):.1f}）")
    print(f"   >>> t=末 跨度      {min(accs_last.values()):.1f} 至 "
          f"{max(accs_last.values()):.1f}")
    if "folora_v2" in series and "ewc" in series:
        fo = series["folora_v2"]["avg_accuracy_seen_so_far"]
        ew = series["ewc"]["avg_accuracy_seen_so_far"]
        d = [a - b for a, b in zip(fo, ew)]
        print(f"   >>> FOLoRA-EWC 逐点差：范围 [{min(d):+.2f}, {max(d):+.2f}]，"
              f"绝对值上限 {max(abs(x) for x in d):.2f}，跨度 {max(d)-min(d):.2f}")
        print(f"       差值在 t=1 / t=5 / t=20 分别为 {d[0]:+.2f} / {d[4]:+.2f} / {d[-1]:+.2f}")
    for other in ("olora", "l2p", "coda", "seq"):
        if other in series and "folora_v2" in series:
            a = series[other]["avg_accuracy_seen_so_far"][0]
            b = series["folora_v2"]["avg_accuracy_seen_so_far"][0]
            print(f"   >>> t=1 处的 {other:10s} = {a:6.1f}（FOLoRA {b:.1f}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="all",
                    choices=["all", "cifar100", "imagenetr"])
    args = ap.parse_args()
    todo = (["cifar100", "imagenetr"] if args.benchmark == "all"
            else [args.benchmark])
    names = {"cifar100": "CIFAR-100", "imagenetr": "ImageNet-R"}
    for b in todo:
        _bench(b, names[b])
        _curves(b, names[b])
    print()


if __name__ == "__main__":
    main()
