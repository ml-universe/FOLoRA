# -*- coding: utf-8 -*-
"""E1：**Fisher 加权 vs 等权**的逐 λ 对照（回应「增益来自分配方式还是正则强度」）。

审稿人的反驳很硬：等权（每条 protected 方向权重相同）在 λ 上只是「需要更大的正则」，
那么 FOLoRA 的增益就来自正则强度、而不是 Fisher 权重本身。判决点因此是**整条 λ 曲线**：
若两臂的差距随 λ 单调收窄并在某个 λ 处消失，反驳成立；若加权臂在每个 λ 上都更好，
或差距在某处重新张开，反驳不成立。

口径
----
* 数据 = `reports/ncm/{bench}/*.json`（NCM，与主表同口径），**不是** results.json。
* 两臂同 seed 配对（`scipy.stats.ttest_rel`）。λ 之间**不做**多重校正：这是同一条曲线的
  四个点，不是一个假说族；需要校正时在正文按单点讨论，不要在这里偷加成全族显著。
* `n` 打成**共同 seed 数**：等权臂的 top-up（`eq_topup_n10`）完成前后 n 会不同，
  读表的人必须能看出这一点，否则「λ=1 的 n=10、λ=30 的 n=5」会被误读成同预算比较。

用法
----
  py -3.11 -m scripts.eq_contrast
  py -3.11 -m scripts.eq_contrast --out reports/eq_contrast.json
"""

import argparse
# Windows 控制台是 GBK：U+2212 之类的字符会让 print 抛 UnicodeEncodeError，
# 把已经算好的结果卡在最后一行。这里降级替换，不让编码问题影响退出码。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
import json
import sys
import statistics as st
from pathlib import Path

from scipy import stats

from scripts.significance import find_key, load_runs_ncm, values

# (λ, 加权 tag, 等权 tag, k)
PAIRS = [
    (1, "v2f_l1_k64", "v2f_eq_l1_k64", 64),
    (3, "v2f_l3_k64", "v2f_eq_l3_k64", 64),
    (10, "v2f_l10_k64", "v2f_eq_l10_k64", 64),
    (30, "v2f_l30_k64", "v2f_eq_l30_k64", 64),
    # 秩匹配臂：等权只有 k=16 时才与加权的 k=16 消融同秩
    (3, "v2f_l3_k16", "v2f_eq_l3_k16", 16),
    (10, "v2f_l10_k16", "v2f_eq_l10_k16", 16),
]
OURS = "folora_v2"


def _rows(grouped, bench: str, metric: str = "final_acc_cil"):
    out = []
    for lam, wt, et, k in PAIRS:
        kw = find_key(grouped, OURS, wt)
        ke = find_key(grouped, OURS, et)
        if kw is None or ke is None:
            out.append({"benchmark": bench, "lambda": lam, "k": k,
                        "status": f"缺臂（加权={kw is not None} 等权={ke is not None}）"})
            continue
        seeds = sorted(set(grouped[kw]) & set(grouped[ke]))
        if len(seeds) < 2:
            out.append({"benchmark": bench, "lambda": lam, "k": k,
                        "status": f"共同 seed 只有 {len(seeds)} 个"})
            continue
        a = values({s: grouped[kw][s] for s in seeds}, metric)
        b = values({s: grouped[ke][s] for s in seeds}, metric)
        d = [x - y for x, y in zip(a, b)]
        m = st.mean(d)
        sd = st.stdev(d) if len(d) > 1 else 0.0
        t, p = stats.ttest_rel(a, b)
        # `values()` 已经把比例换成百分数（×100），这里**不能再乘一次**
        out.append({
            "benchmark": bench, "lambda": lam, "k": k, "n": len(seeds),
            "weighted_mean": round(st.mean(a), 2),
            "equal_mean": round(st.mean(b), 2),
            "delta": round(m, 2),
            "sd_of_diff": round(sd, 2),
            "t": round(float(t), 2), "p": round(float(p), 4),
            "seeds": seeds,
        })
    return out


def _arm_curves(grouped, metric: str = "final_acc_cil"):
    """每一臂**各自的** λ 曲线（不配对、不看另一臂在不在）。

    为什么必须单独打：本文 §5.5 的判决句是「等权臂自己在 λ=10 就爬到 77.86、λ=30 仍是
    77.81，因此两臂平台相同」——那句里的 77.81 只有**等权**臂有，`_rows()` 因为
    加权 λ=30 缺席而整行跳过，这个数字就无处可查。凡正文引用的数都必须能在这里追到。
    """
    out = []
    for arm, pat in (("weighted", "v2f_l{}_k64"), ("equal", "v2f_eq_l{}_k64")):
        for lam in (1, 3, 10, 30, 100, 300):
            k = find_key(grouped, OURS, pat.format(lam))
            if k is None:
                continue
            v = values(grouped[k], metric)
            out.append({"arm": arm, "lambda": lam, "k": 64, "n": len(v),
                        "mean": round(st.mean(v), 2),
                        "sd": round(st.stdev(v), 2) if len(v) > 1 else None,
                        "seeds": sorted(grouped[k])})
    return out


def _best_tuned(grouped, metric: str = "final_acc_cil"):
    """各臂在自己的最优 λ 上对打（R1-1 明确要求的那一种比较）。

    为什么要单列：匹配 λ 的比较会让「另一臂只是没调够」这句话永远说不完；只有在各自
    最优处比较，才能回答「调好之后谁高」。代价是 λ 由数据选出来（post-hoc），所以
    产物里把两边选中的 λ 一并落盘，正文也必须写明这是探索性比较。

    **配对仍然只取共同 seed**：两臂的最优 λ 不同，但 seed 是同一个，逐 seed 相减才
    抵消初始化差异；跨 seed 相减会把差异鼓成噪声。
    """
    def best(pat):
        cand = []
        for lam in (1, 3, 10, 30, 100, 300):
            k = find_key(grouped, OURS, pat.format(lam))
            if k is None:
                continue
            v = values(grouped[k], metric)
            if len(v) >= 2:
                cand.append((st.mean(v), lam, k))
        return max(cand, key=lambda t: t[0]) if cand else None

    bw, be = best("v2f_l{}_k64"), best("v2f_eq_l{}_k64")
    if bw is None or be is None:
        return None
    seeds = sorted(set(grouped[bw[2]]) & set(grouped[be[2]]))
    if len(seeds) < 2:
        return {"status": f"共同 seed 只有 {len(seeds)} 个",
                "weighted_lambda": bw[1], "equal_lambda": be[1]}
    a = values({s: grouped[bw[2]][s] for s in seeds}, metric)
    b = values({s: grouped[be[2]][s] for s in seeds}, metric)
    d = [x - y for x, y in zip(a, b)]
    t, p = stats.ttest_rel(a, b)
    return {"weighted_lambda": bw[1], "equal_lambda": be[1], "n": len(seeds),
            "weighted_mean": round(st.mean(a), 2), "equal_mean": round(st.mean(b), 2),
            "delta": round(st.mean(d), 2),
            "sd_of_diff": round(st.stdev(d), 2) if len(d) > 1 else None,
            "t": round(float(t), 2), "p": round(float(p), 4), "seeds": seeds}


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 加权 vs 等权逐 λ 对照")
    ap.add_argument("--benchmark", default="cifar100")
    ap.add_argument("--metric", default="final_acc_cil")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grouped = load_runs_ncm(Path("reports/ncm"), args.benchmark)
    rows = _rows(grouped, args.benchmark, args.metric)
    curves = _arm_curves(grouped, args.metric)
    bt = _best_tuned(grouped, args.metric)

    print(f"=== {args.benchmark}: Fisher 加权 vs 等权（同一 λ、同一 seed 配对）===")
    print(f"{'lambda':>6s} {'k':>3s} {'n':>3s} {'weighted':>9s} {'equal':>9s} "
          f"{'delta':>7s} {'sd':>6s} {'t':>7s} {'p':>8s}")
    print("-" * 66)
    for r in rows:
        if "status" in r:
            print(f"{r['lambda']:>6} {r['k']:>3}  --  {r['status']}")
            continue
        star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else \
            "*" if r["p"] < 0.05 else ""
        print(f"{r['lambda']:>6} {r['k']:>3} {r['n']:>3} {r['weighted_mean']:>9.2f} "
              f"{r['equal_mean']:>9.2f} {r['delta']:>+7.2f} {r['sd_of_diff']:>6.2f} "
              f"{r['t']:>7.2f} {r['p']:>8.4f} {star}")

    print(f"\n=== 每臂各自的 λ 曲线（k=64；不配对，缺臂不影响这一块）===")
    print(f"{'arm':>9s} {'lambda':>6s} {'n':>3s} {'mean':>8s} {'sd':>6s}")
    for c in curves:
        sd = "--" if c["sd"] is None else f"{c['sd']:.2f}"
        print(f"{c['arm']:>9s} {c['lambda']:>6} {c['n']:>3} {c['mean']:>8.2f} {sd:>6s}")

    if bt:
        print("\n=== 各自最优 λ 上的对打（R1-1 要求的那种比较；λ 由数据选出，属探索性）===")
        if "status" in bt:
            print(f"   {bt['status']}")
        else:
            print(f"   加权 λ={bt['weighted_lambda']} {bt['weighted_mean']:.2f}  vs  "
                  f"等权 λ={bt['equal_lambda']} {bt['equal_mean']:.2f}   "
                  f"Δ={bt['delta']:+.2f}±{bt['sd_of_diff']:.2f} "
                  f"t={bt['t']:+.2f} p={bt['p']:.4f} n={bt['n']}")

    ok = [r for r in rows if "delta" in r]
    if ok:
        # 判决句只报告数据说了什么，不预先假定方向
        mono = all(x["delta"] >= y["delta"] for x, y in zip(ok[:4], ok[1:4]))
        print(f"\nΔ = 加权 - 等权（正 = 加权更好）。k=64 的三个/四个 λ 上 Δ 单调收窄：{mono}")
        pos = sum(1 for r in ok if r["delta"] > 0)
        print(f"Δ>0 的点：{pos}/{len(ok)}；显著（p<0.05）的点："
              f"{sum(1 for r in ok if r['p'] < 0.05)}/{len(ok)}")
        print("注：**不要**只说「大部分为正」。若某 λ 上 Δ 不显著，正文必须写不显著，"
              "\n    并说明它是否落在 0.89 点的单次比较分辨率内（reports/repro_floor.json）。")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"benchmark": args.benchmark, "metric": args.metric,
                   "rows": rows, "arm_curves": curves, "best_tuned": bt},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
