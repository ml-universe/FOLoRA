# -*- coding: utf-8 -*-
"""每一行 NCM 精度对**冻结特征地板**的单样本检验（Sec. 5.2 / 回复信 R8-4(iii)）。

为什么要单独落一份 JSON：正文里写着「O-LoRA 比地板低 5.4 分，p=0.0003；
Seq-LoRA 低 1.0 分，p=0.13」，而这两个 p 值原先只在会话里算过、没有落到 `reports/`。
按回复信的规矩（信里每个数都要能在 `reports/` 里查到），必须可复算。

口径
----
* 地板 = SimpleCIL（冻结主干 + NCM），**确定性**，n=1、无方差，因此检验是
  「样本 vs 常数」的单样本 t 检验，不是两样本。
* 逐 seed 的 NCM 精度取自 `reports/ncm/{bench}/*.json`（与主表同源）。
* 同时给双尾 p 与「低于地板」的单尾 p。正文用的是双尾，因为它不预设方向；
  单尾也一并落盘，供将来引用时不再重算。

用法
----
    py -3.11 -m scripts.floor_tests
"""

import glob
import json
import statistics as st
from pathlib import Path

from scipy import stats

# 与 `paper/tables/tab_main.tex` 的行一一对应；tag 必须和主表相同
ROWS = [
    ("seq", "default", "Seq-LoRA"),
    ("ewc", None, "EWC-LoRA (tuned)"),      # tag 逐基准不同，下面特判
    ("olora", "default_fix1", "O-LoRA"),
    ("inflora", "default_fix1", "InfLoRA"),
    ("l2p", "pilot20", "L2P"),
    ("coda", "pool100_len8_ep20", "CODA-Prompt"),
    ("folora_v2", "v2f_l3_k64", "FOLoRA"),
    # §5.6 的两个「训练时正交项」强度：正文说它们把 O-LoRA 拉回地板之上，
    # 那就必须和主表行用同一把尺子量，否则「above the floor」在别处是检验、
    # 在这里是目测。
    ("olora", "olora_orth_l0.1_fix1", "O-LoRA + orth ($\\lambda_1$=0.1)"),
    ("olora", "olora_orth_l1_fix1", "O-LoRA + orth ($\\lambda_1$=1.0)"),
]
EWC_TAG = {"cifar100": "ewc_lam300", "imagenetr": "ewc_lam1000"}
BENCHES = ["cifar100", "imagenetr"]


def load(bench):
    out = {}
    for p in glob.glob(f"reports/ncm/{bench}/*.json"):
        d = json.load(open(p, encoding="utf-8"))
        out[(d.get("method"), d.get("tag"), int(d.get("seed", -1)))] = \
            d["final_acc_cil"] * 100
    return out


def main():
    report = {}
    for bench in BENCHES:
        ncm = load(bench)
        floor = ncm.get(("simplecil", "frozen", 0))
        if floor is None:
            raise SystemExit(f"{bench}: 缺 SimpleCIL 地板，先跑 eval_ncm_sweep")
        rows = {}
        for method, tag, label in ROWS:
            t = EWC_TAG[bench] if method == "ewc" else tag
            vals = sorted(v for (m, tg, s), v in ncm.items() if m == method and tg == t)
            if len(vals) < 2:
                rows[label] = {"n": len(vals), "skipped": "seed 少于 2，无法检验"}
                continue
            delta = st.mean(vals) - floor
            tt, pp = stats.ttest_1samp(vals, floor)      # 双尾
            pp_below = stats.t.cdf(tt, len(vals) - 1)    # 单尾：低于地板
            rows[label] = {
                "n": len(vals),
                "mean": round(st.mean(vals), 3),
                "sd": round(st.pstdev(vals), 3),
                "floor": floor,
                "delta": round(delta, 3),
                "t": round(float(tt), 3),
                "p_two_sided": round(float(pp), 6),
                "p_below": round(float(pp_below), 6),
                "seeds": len(vals),
            }
        report[bench] = rows
        print(f"== {bench} 地板 {floor:.2f}")
        for label, r in rows.items():
            if "skipped" in r:
                print(f"   {label:22s} 跳过：{r['skipped']}（n={r['n']}）")
            else:
                print(f"   {label:22s} n={r['n']:2d} Δ={r['delta']:+6.2f} "
                      f"t={r['t']:+6.2f} p(双尾)={r['p_two_sided']:.4f} "
                      f"p(低于)={r['p_below']:.4f}")
    Path("reports/floor_tests.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写入 reports/floor_tests.json")


if __name__ == "__main__":
    main()
