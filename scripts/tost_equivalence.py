r"""TOST 等价检验：把「FOLoRA 与 EWC-LoRA 打平」从「没检出差异」升格为「差异小到可忽略」。

为什么需要它（回应 R1 / R8 关于「统计上不显著 ≠ 等价」的那条）
--------------------------------------------------------------
主表里我们对 EWC-LoRA 是**互有胜负**：CIFAR-100 上 $\lambda{=}300$ 的 ACC 比我们高
0.13 点（$p=0.64$），ImageNet-R 上 $\lambda{=}1000$ 的 ACC 比我们低 0.04 点（$p=0.92$）。
「配对 t 检验没有拒绝原假设」只说明**没检出差异**——样本量小、噪声大时，这句话与
「两者其实等价」并不等义，审稿人完全可以反问「你只是没跑够种子」。TOST 反过来：
把「差异超过 δ」作为原假设，**拒绝**它才是正面结论（等价于在 δ 范围内证明了相等）。

判据
----
给定配对差 $d_i = a_i - b_i$（$a$ = FOLoRA，$b$ = 基线），自由度 $n-1$：
  下界检验  $H_0:\mu\le-\delta$，$t_1=(\bar d+\delta)/s_{\bar d}$，$p_1=1-F(t_1)$
  上界检验  $H_0:\mu\ge+\delta$，$t_2=(\bar d-\delta)/s_{\bar d}$，$p_2=F(t_2)$
  $p_{\mathrm{TOST}}=\max(p_1,p_2)$，$p_{\mathrm{TOST}}<\alpha$ ⇒ 在 $\pm\delta$ 内等价。

δ 怎么定（**必须事先定，不能看着结果挑**）
------------------------------------------
默认 $\delta=1.0$ 个百分点，理由是同 seed 同配置两次独立 run 之差的 sd 实测为 0.89 点
（`reports/repro_floor.json`，E4：单次 run 的 sd 0.63 点 × √2，共 11 条重复 run）。
1.0 点比这个「单次比较的分辨率」略宽——比它更小的差异在任何单次实验里都测不出来，
属于「可忽略」的合理定义。**δ 不得小于 0.89**：小于噪声地板的等价结论是跑多少 seed
都立不住的。`--delta` 可改，但改了就**必须在正文写明 δ 及其理由**。
（2026-11-04 核对：δ=0.89 时两个基准的调优点仍然全部通过 TOST，
故 δ=1.0 的旧结论不依赖这个选择。）
另附 $\delta=0.5$（更严）作为敏感性分析一并打印。

用法
----
  py -3.11 -m scripts.tost_equivalence
  py -3.11 -m scripts.tost_equivalence --delta 0.5 --benchmark imagenetr
  py -3.11 -m scripts.tost_equivalence --out reports/tost_cifar100.json
"""

import argparse
import json
from pathlib import Path

from scipy import stats

from scripts.significance import FIX1, find_key, load_runs_ncm, values

METRICS = [("final_acc_cil", "ACC"), ("forgetting_cil", "GAP")]

# (基线 method, tag, 人读标签)；与主表行一一对应
BASELINES = [
    ("seq", "default", "Seq-LoRA"),
    ("ewc", "ewc_lam300", "EWC-LoRA (lam=300)"),
    ("ewc", "ewc_lam1000", "EWC-LoRA (lam=1000)"),
    # 预注册的判据基线 λ=100 在缓存里就叫 `default`（10 seed）；`ewc_lam100` 只有 1 条
    ("ewc", "default", "EWC-LoRA (lam=100)"),
    # O-LoRA / InfLoRA 必须走 `_fix1`（旧 tag 的 checkpoint 只存末个 adapter，
    # 「评估的模型 ≠ 训练的模型」）；两者共用 significance.FIX1 这一个开关，
    # 由 tests/test_fix1_tag_consistency.py 守着，不能只切一半。
    ("olora", f"default{FIX1}", "O-LoRA"),
    ("inflora", f"default{FIX1}", "InfLoRA"),
    ("l2p", "pilot20", "L2P"),                       # prompt 类 = 20 任务 × 20 epoch
    ("coda", "pool100_len8_ep20", "CODA-Prompt"),
]

OURS = ("folora_v2", "v2f_l3_k64")


def tost(a, b, delta: float):
    """配对 TOST。返回 (n, mean_diff, se, p_lower, p_upper, p_tost)。"""
    n = min(len(a), len(b))
    if n < 2:
        return n, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    d = [a[i] - b[i] for i in range(n)]
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = (var / n) ** 0.5
    if se == 0:
        # 逐位相同的差（例如两边都是确定性基线）：等价与否完全由 |m| < delta 决定
        p = 0.0 if abs(m) < delta else 1.0
        return n, m, 0.0, p, p, p
    df = n - 1
    t1 = (m + delta) / se
    t2 = (m - delta) / se
    p1 = 1.0 - stats.t.cdf(t1, df)     # H0: mu <= -delta
    p2 = stats.t.cdf(t2, df)           # H0: mu >= +delta
    return n, m, se, p1, p2, max(p1, p2)


def main() -> None:
    ap = argparse.ArgumentParser(description="TOST 等价检验（对主表每一行）")
    ap.add_argument("--benchmark", default="cifar100")
    ap.add_argument("--cache_root", default="reports/ncm")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="等价边界（百分点）。默认 1.0，理由见模块 docstring")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grouped = load_runs_ncm(Path(args.cache_root), args.benchmark)
    k_ours = find_key(grouped, *OURS)
    if k_ours is None:
        raise SystemExit(f"NCM 缓存里找不到我们的主表行 {OURS}")

    rows = []
    for method, tag, label in BASELINES:
        k = find_key(grouped, method, tag)
        if k is None:
            print(f"[跳过] {label}: 缓存里没有 {method}/{tag}")
            continue
        seeds = sorted(set(grouped[k_ours]) & set(grouped[k]))
        if len(seeds) < 2:
            print(f"[跳过] {label}: 共同 seed 只有 {len(seeds)} 个")
            continue
        for metric, mlabel in METRICS:
            a = values({s: grouped[k_ours][s] for s in seeds}, metric)
            b = values({s: grouped[k][s] for s in seeds}, metric)
            n, m, se, p1, p2, p = tost(a, b, args.delta)
            _, m5, _, _, _, p5 = tost(a, b, 0.5)
            rows.append({
                "baseline": label, "method": method, "tag": tag, "metric": mlabel,
                "n_paired_seeds": n, "mean_diff_pts": round(m, 3),
                "se_pts": round(se, 3), "delta": args.delta,
                "p_lower": round(p1, 5), "p_upper": round(p2, 5),
                "p_tost": round(p, 5),
                "equivalent": bool(p < args.alpha),
                "p_tost_delta_0.5": round(p5, 5),
                "equivalent_delta_0.5": bool(p5 < args.alpha),
            })

    print(f"\n=== {args.benchmark}: TOST 等价检验（delta={args.delta} 点, alpha={args.alpha}）===")
    print(f"{'baseline':22s} {'metric':6s} {'n':>3s} {'diff':>8s} {'se':>6s} "
          f"{'p_TOST':>9s} {'等价?':>6s}  {'p(δ=0.5)':>10s}")
    print("-" * 84)
    for r in rows:
        mark = "是" if r["equivalent"] else "否"
        print(f"{r['baseline']:22s} {r['metric']:6s} {r['n_paired_seeds']:3d} "
              f"{r['mean_diff_pts']:+8.2f} {r['se_pts']:6.2f} {r['p_tost']:9.4f} "
              f"{mark:>6s}  {r['p_tost_delta_0.5']:10.4f}")

    print("\n注：ACC 为正表示我们更高；GAP 为负表示我们的落差更小。"
          "\n    「等价」= 在 ±δ 内证明了相等，不是「没检出差异」。")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"benchmark": args.benchmark, "delta": args.delta,
                   "alpha": args.alpha, "rows": rows},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
