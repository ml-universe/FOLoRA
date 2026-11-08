# -*- coding: utf-8 -*-
"""把本轮大修新增的几组对照**落成一份可核验的 json**（原本只在终端里算过，没有底稿）。

为什么要单独写这一个脚本
------------------------
L2-SP 打平、随机子空间、界项探针、T=50 长序列这四组数字，本轮是临时算出来的，
结果没有像 `eq_contrast.json` 那样落盘。**没有落盘的数字在凭证文件里就是无源之水**——
几个月后既无法复算，也无法回答「这个 p 值是怎么来的」。所以这里重算一遍并写死。

口径与 `eq_contrast.py` 一致
---------------------------
* 数据 = `reports/ncm/<bench>/*.json`（NCM，与主表同口径），**不是** results.json。
* 同 seed 配对，`scipy.stats.ttest_rel`（双侧）。
* 共同 seed 数 < 2 时**不做检验**，只记录均值与 n —— n=1 时任何 p 值都是编造。

用法
----
    py -3.11 -m scripts.revision_extra_tests --out reports/revision_extra_tests.json
"""

import argparse
import json
import statistics as st
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# (标签, 基准, 方法A, tagA, 方法B, tagB)
CONTRASTS = [
    ("A4 L2-SP λ=3 vs FOLoRA λ=3", "cifar100", "folora_v2", "v2f_l2sp_l3",
     "folora_v2", "v2f_l3_k64"),
    ("A4 L2-SP λ=10 vs FOLoRA λ=10", "cifar100", "folora_v2", "v2f_l2sp_l10",
     "folora_v2", "v2f_l10_k64"),
    ("E7 随机子空间 λ=3 vs FOLoRA λ=3", "cifar100", "folora_v2", "v2f_rand_l3_k64",
     "folora_v2", "v2f_l3_k64"),
    ("A5 界项探针 λ=3 vs FOLoRA λ=3", "cifar100", "folora_v2", "bound_l3_k64",
     "folora_v2", "v2f_l3_k64"),
    ("T=50 FOLoRA λ=10 vs EWC λ=1000", "cifar100", "folora_v2", "t50_l10_k64",
     "ewc", "t50_ewc_l1000"),
    ("A5 界项探针 λ=3 (ImageNet-R)", "imagenetr", "folora_v2", "bound_l3_k64",
     "folora_v2", "v2f_l3_k64"),
]


def load(bench: str) -> dict:
    """{(method, tag): {seed: acc%}}，直接读缓存，绕开任何协议过滤。"""
    out = {}
    for p in sorted((ROOT / "reports" / "ncm" / bench).glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            out.setdefault((j["method"], j["tag"]), {})[int(j["seed"])] = \
                j["final_acc_cil"] * 100
        except Exception:
            continue
    return out


def describe(vals: list) -> dict:
    n = len(vals)
    return {"n": n, "mean": round(st.mean(vals), 2),
            "sd": round(st.stdev(vals), 2) if n > 1 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/revision_extra_tests.json")
    args = ap.parse_args()

    cache = {b: load(b) for b in ("cifar100", "imagenetr")}
    rows = []
    for label, bench, ma, ta, mb, tb in CONTRASTS:
        g = cache[bench]
        a = g.get((ma, ta), {})
        b = g.get((mb, tb), {})
        seeds = sorted(set(a) & set(b))
        row = {"contrast": label, "benchmark": bench,
               "arm_a": f"{ma}__{ta}", "arm_b": f"{mb}__{tb}",
               "a": describe([a[s] for s in seeds]) if seeds else None,
               "b": describe([b[s] for s in seeds]) if seeds else None,
               "seeds": seeds}
        if len(seeds) >= 2:
            xa = [a[s] for s in seeds]
            xb = [b[s] for s in seeds]
            d = [x - y for x, y in zip(xa, xb)]
            t, p = stats.ttest_rel(xa, xb)
            row.update({"n": len(seeds), "delta": round(st.mean(d), 2),
                        "sd_of_diff": round(st.stdev(d), 2),
                        "t": round(float(t), 2), "p": round(float(p), 4)})
        else:
            row["status"] = f"共同 seed 只有 {len(seeds)} 个，不做检验（n<2 的 p 值无意义）"
        rows.append(row)

    out = {"benchmark": "cifar100/imagenetr", "metric": "final_acc_cil",
           "note": "同 seed 配对双侧 t 检验；数据来自 reports/ncm/*.json，×100 为百分数",
           "rows": rows}
    path = ROOT / args.out
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in rows:
        if "p" in r:
            print(f"{r['contrast']:<40s} n={r['n']} A={r['a']['mean']:.2f} "
                  f"B={r['b']['mean']:.2f} Δ={r['delta']:+.2f} t={r['t']:+.2f} p={r['p']:.4f}")
        else:
            print(f"{r['contrast']:<40s} {r['status']}")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
