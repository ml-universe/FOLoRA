"""学习头的「任务新近度分层」统计（零训练，回应 R8-4）。

为什么需要它
------------
R8-4 质疑 Table 2 的学习头整体只有 7--9%：「这说明是分类器坏了（任务新近度偏置），
不是预算不够」。论文的回应不是修一个偏差校正头，而是给出诊断：把终态模型的学习头
准确率按**任务新近度**分成三档——最后一个任务、倒数第二个、其余十八个——如果是
头把决策边界挪到了最近的类上，三档应呈单调递减、且大部分任务低于 1% 随机水平。

数据来源
--------
`experiments/<bench>/<method>/<tag>/seed<k>/results.json` 的 `acc_cil[-1]`：训练器在
最后一个任务边界上记录的**逐任务学习头准确率**（CIL 口径，即 100 维全头 argmax 后
只取该任务类那一段）。这与 Table 2 的 Head 栏（`ncm_summary_*.json` 的
`head_acc_mean`）是同一个量，`all` 一列应与 Table 2 对得上——本脚本会核对这一点。

用法
----
    py -3.11 -m scripts.head_recency_tiers                # 写 reports/head_recency_tiers.json
    py -3.11 -m scripts.head_recency_tiers --check        # 只核对，不写（漂移即非零退出）
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"

# Table 2 的行，(标签, method, cifar tag, imagenetr tag)；冻结特征地板没有训练头，不在此列。
ROWS = [
    ("Seq-LoRA", "seq", "default", "default"),
    ("EWC-LoRA", "ewc", "ewc_lam300", "ewc_lam1000"),
    ("O-LoRA", "olora", "olora_orth_l1_fix1", "olora_orth_l1_fix1"),
    ("L2P", "l2p", "pilot20", "pilot20"),
    ("CODA-Prompt", "coda", "pool100_len8_ep20", "pool100_len8_ep20_inr"),
    ("InfLoRA", "inflora", "default_fix1", "default_fix1"),
    ("FOLoRA (ours)", "folora_v2", "v2f_l3_k64", "v2f_l3_k64"),
]


def per_seed(bench: str, method: str, tag: str) -> list[tuple[int, list[float]]]:
    out = []
    for d in sorted((EXPERIMENTS / bench / method / tag).glob("seed*")):
        rp = d / "results.json"
        if not rp.exists():
            continue
        d0 = json.loads(rp.read_text(encoding="utf-8"))
        final = d0["acc_cil"][-1]
        out.append((int(d.name.replace("seed", "")), [x * 100 for x in final]))
    return out


def tier_stats(runs: list[tuple[int, list[float]]]) -> dict:
    # 先按 seed 取平均（与论文「同一批 seed 上平均」一致），再分层。
    mean = [st.mean(r[i] for _, r in runs) for i in range(len(runs[0][1]))]
    return {
        "n_seeds": len(runs),
        "seeds": [s for s, _ in runs],
        "last": round(mean[-1], 2),
        "second_last": round(mean[-2], 2),
        # 论文引的是「除最后两个任务之外」的平均（last / second_last / first_18 三档
        # 正好划分 20 个任务）；first_19 只是留档，曾把这两个搞混过一次（4.2 是
        # first_19，被误标成「前十八」）。
        "first_18": round(st.mean(mean[:-2]), 2),
        "first_19": round(st.mean(mean[:-1]), 2),
        "all": round(st.mean(mean), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="学习头的新近度分层统计（零训练）")
    ap.add_argument("--check", action="store_true", help="只核对 reports/head_recency_tiers.json，不写")
    ap.add_argument("--out", default="reports/head_recency_tiers.json")
    args = ap.parse_args()

    result: dict = {"metric": "learned-head per-task accuracy at the final boundary (results.json acc_cil[-1], x100)",
                    "benchmarks": {}}
    missing = []
    for bench in ("cifar100", "imagenetr"):
        result["benchmarks"][bench] = {}
        for label, method, t_c, t_i in ROWS:
            tag = t_c if bench == "cifar100" else t_i
            runs = per_seed(bench, method, tag)
            if not runs:
                missing.append(f"{bench}/{method}/{tag}")
                continue
            result["benchmarks"][bench][label] = {"method": method, "tag": tag,
                                                  **tier_stats(runs)}

    if missing:
        raise SystemExit(f"缺 run（tag 打错？）：{missing}")

    # 与 Table 2 的 Head 栏对账：all 应等于 ncm_summary 的 head_acc_mean（两位小数内）。
    drift = []
    for bench in ("cifar100", "imagenetr"):
        summ = json.loads((REPORTS / f"ncm_summary_{bench}.json").read_text(encoding="utf-8"))
        for label, rec in result["benchmarks"][bench].items():
            key = f"{rec['method']}/{rec['tag']}"
            if key not in summ:
                continue
            ref = summ[key]["head_acc_mean"] * 100
            if abs(ref - rec["all"]) > 0.02:
                drift.append(f"{bench} {label}: tiers all={rec['all']:.2f} vs Table2={ref:.2f}")

    out = Path(args.out)
    if args.check:
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        if old != result:
            print("reports/head_recency_tiers.json 与现算不一致 —— 重跑不带 --check 以更新")
            sys.exit(1)
        print("head_recency_tiers.json 一致")
    else:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"写 {out}")

    for bench in ("cifar100", "imagenetr"):
        print(f"\n{bench}")
        for label, rec in result["benchmarks"][bench].items():
            print(f"  {label:14s} n={rec['n_seeds']:2d}  last={rec['last']:5.1f}  "
                  f"2nd={rec['second_last']:5.1f}  first18={rec['first_18']:4.1f}  "
                  f"first19={rec['first_19']:4.1f}  all={rec['all']:4.1f}")
    if drift:
        print("\n与 Table 2 的对账不一致：")
        for d in drift:
            print("  " + d)
        sys.exit(1)


if __name__ == "__main__":
    main()
