# -*- coding: utf-8 -*-
"""O-LoRA「训练时正交项」两个强度对 FOLoRA 的配对检验（Sec. 5.6 / R1、R8）。

正文 §5.6 写着「在这些更强的配置面前，FOLoRA 的优势从 +12.92 收窄到 +7.10 和
+5.60（CIFAR-100）…… 家族内 Holm 校正后仍显著」。这些数原先只在会话里算过，
`reports/` 里查不到 —— 按回复信的规矩（信里每个数都要能回溯）必须可复算。

口径
----
* 数据取 `reports/ncm/{bench}/*.json`，与主表同源（NCM，非 head-acc）。
* **配对**：FOLoRA 与 O-LoRA-orth 用同一个 class-order seed，逐 seed 相减再检验；
  跨 seed 相减会把显著性问题化成噪声（`reports/repro_floor.json`：同配置两次独立
  run 的差 sd ≈ 0.89 点）。
* **seed 预算逐基准不同**：CIFAR-100 的 orth 变体已补到 5 个种子，ImageNet-R 仍 3 个。
  脚本按实际存在的交集取 n，并把 n 写进产物，正文照抄，不允许一处写 5、一处写 3。
* **多重比较**：两个强度构成一个两成员家族，用 Holm step-down（不是 Bonferroni）。
  两者的区别对本例是实打实的：家族里较大的那个 p 乘数降为 1，所以 Bonferroni 的
  0.0259 与 Holm 的 0.0129 不等价，正文印哪个必须与这里一致。

用法
----
    py -3.11 -m scripts.orth_variant_tests
"""

import glob
import json
import statistics as st
from pathlib import Path

from scipy import stats

OURS = ("folora_v2", "v2f_l3_k64")
VARIANTS = ["olora_orth_l0.1_fix1", "olora_orth_l1_fix1"]
BENCHES = ["cifar100", "imagenetr"]


def load(bench, method, tag):
    out = {}
    for p in glob.glob(f"reports/ncm/{bench}/*.json"):
        d = json.load(open(p, encoding="utf-8"))
        if d.get("method") == method and d.get("tag") == tag:
            out[int(d["seed"])] = d["final_acc_cil"] * 100
    return out


def holm(ps):
    """Holm step-down：返回与输入同序的校正 p。"""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (len(ps) - rank) * ps[i])
        v = max(v, prev)
        adj[i] = v
        prev = v
    return adj


def main():
    report = {}
    for bench in BENCHES:
        ours = load(bench, *OURS)
        rows, raws = [], []
        for tag in VARIANTS:
            o = load(bench, "olora", tag)
            seeds = sorted(set(ours) & set(o))
            if len(seeds) < 2:
                continue
            a = [ours[s] for s in seeds]
            b = [o[s] for s in seeds]
            d = [x - y for x, y in zip(a, b)]
            t, p = stats.ttest_rel(a, b)
            rows.append({
                "variant": tag,
                "n": len(seeds),
                "seeds": seeds,
                "variant_mean": round(st.mean(b), 3),
                "variant_sd": round(st.stdev(b), 3),
                "delta": round(st.mean(d), 3),
                "sd_of_diff": round(st.stdev(d), 3),
                "t": round(float(t), 3),
                "p": round(float(p), 6),
            })
            raws.append(float(p))
        for row, ph in zip(rows, holm(raws)):
            row["p_holm"] = round(ph, 6)
        report[bench] = rows

        print(f"== {bench}")
        for r in rows:
            print(f"   {r['variant']:22s} n={r['n']} "
                  f"对照 {r['variant_mean']:.2f}±{r['variant_sd']:.2f}  "
                  f"Δ={r['delta']:+.2f}±{r['sd_of_diff']:.2f} "
                  f"t={r['t']:+.2f} p={r['p']:.5f} p_holm={r['p_holm']:.5f}")

    Path("reports/orth_variant_tests.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写入 reports/orth_variant_tests.json")


if __name__ == "__main__":
    main()
