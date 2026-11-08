# -*- coding: utf-8 -*-
"""生成 `paper/tables/tab_bwt.tex`：**真实时间遗忘**表（E2）。

数据源是 `reports/forgetting_true_cifar100.json`（`scripts/forgetting_analysis.py
--preset e2b` 的输出），**不是** `reports/ncm_summary_*.json`。两者的「遗忘」不是同一个
量：前者来自逐任务边界重提特征（θ_t 提特征、A[t][j]），后者来自终态模型重打分
（候选集扩张造成的落差，本文叫 GAP）。混用会让表里出现一个无法解释的数字。

配对检验在**同一个 seed** 上做（`scipy.stats.ttest_rel`），不是把两组的均值相减：
本仓库同 seed 同配置的两次独立 run 之差，sd 约 0.89 点（`reports/repro_floor.json`），
不配对会严重高估显著性。

用法：
    py -3.11 -m scripts.make_bwt_table [--check]
"""

import argparse
import json
from pathlib import Path

from scipy import stats

SRC = Path("reports/forgetting_true_cifar100.json")
OUT = Path("paper/tables")

# 显示名 → 结论文档里的键。顺序按 BWT 从好到差排（读者第一眼看到的是排序）。
ROWS = [
    ("EWC-LoRA ($\\lambda{=}1000$)", "ewc/ncm_trace_l1000"),
    ("EWC-LoRA ($\\lambda{=}300$, tuned)", "ewc/ncm_trace_l300_trace"),
    ("FOLoRA (ours)", "folora_v2/ncm_trace_l3_k64"),
    ("Seq-LoRA", "seq/ncm_trace"),
]


def _paired(a, b, key):
    """同一 seed 上的配对差与双侧 t 检验。返回 (mean, sd_of_diff, t, p, n)。"""
    da = {e["seed"]: e["trace"][key] for e in a["per_seed"] if e.get("trace")}
    db = {e["seed"]: e["trace"][key] for e in b["per_seed"] if e.get("trace")}
    seeds = sorted(set(da) & set(db))
    xs = [da[s] for s in seeds]
    ys = [db[s] for s in seeds]
    t, p = stats.ttest_rel(xs, ys)
    diffs = [x - y for x, y in zip(xs, ys)]
    m = sum(diffs) / len(diffs)
    sd = (sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    return m, sd, t, p, len(diffs)


def build(check_only=False):
    d = json.load(open(SRC, encoding="utf-8"))
    for _, key in ROWS:
        if key not in d or "bwt" not in d[key]:
            raise SystemExit(f"{SRC} 里没有 {key} 的 bwt —— 先跑 forgetting_analysis --preset e2b")

    ours = d["folora_v2/ncm_trace_l3_k64"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        # 六列（加 AIA 后）在 review 版心 390pt 里溢出 48pt：降一级字号 + 收窄列距。
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{True temporal forgetting on Split CIFAR-100. Unlike the CIL--TIL gap in "
        r"Table~\ref{tab:main}, every entry here comes from a feature extraction at each of the "
        r"20 task boundaries, so $A_{t,j}$ is the accuracy measured with the model as it stood "
        r"\emph{then}. BWT $=\frac{1}{T-1}\sum_{j<T-1}(A_{T-1,j}-A_{j,j})$; the peak-drop "
        r"measure adds a penalty when a task's accuracy falls after it was learned and later "
        r"recovers. Negative BWT means forgetting; both are in percentage points, "
        r"mean$\pm$sd over three seeds. AIA is the average incremental accuracy, the mean over "
        r"boundaries of the accuracy the model then had on every class seen so far; final "
        r"accuracy is from the same runs, so the last three columns are directly comparable "
        r"within a row but are not the 10-seed means of "
        r"Table~\ref{tab:main}; $\lambda{=}300$ is the tuned EWC-LoRA row of that table.}",
        r"\label{tab:bwt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuration & BWT $\downarrow$ & Peak drop $\downarrow$ & AIA $\uparrow$ & Final acc. & $n$ \\",
        r"\midrule",
    ]
    for label, key in ROWS:
        r = d[key]
        lines.append(
            f"{label} & ${r['bwt']['mean']:+.2f}\\pm{r['bwt']['sd']:.2f}$ & "
            f"${r['fgt_max']['mean']:+.2f}\\pm{r['fgt_max']['sd']:.2f}$ & "
            f"${r['aia']['mean']:.2f}\\pm{r['aia']['sd']:.2f}$ & "
            f"${r['acc_final']['mean']:.2f}\\pm{r['acc_final']['sd']:.2f}$ & {r['bwt']['n']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)

    # 证明句里的每一个数都在这里现算，避免手抄
    stats_out = {}
    for name, key in [("ewc300", "ewc/ncm_trace_l300_trace"),
                      ("ewc1000", "ewc/ncm_trace_l1000"),
                      ("seq", "seq/ncm_trace")]:
        for metric in ("bwt", "aia", "acc_final"):
            m, sd, t, p, n = _paired(ours, d[key], metric)
            stats_out[f"ours_vs_{name}_{metric}"] = {"mean": round(m, 2),
                                                     "sd": round(sd, 2),
                                                     "t": round(t, 2),
                                                     "p": round(p, 4), "n": n}
    return text, stats_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text, st = build(args.check)
    print(text)
    print("\n配对检验（FOLoRA − 对照，同一 seed）：")
    for k, v in st.items():
        print(f"  {k:24s} Δ={v['mean']:+.2f}±{v['sd']:.2f}  t={v['t']:+.2f}  p={v['p']:.4f}  n={v['n']}")
    if args.check:
        print("\n[--check] 未写文件")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_bwt.tex").write_text(text + "\n", encoding="utf-8")
    json.dump(st, open("reports/bwt_paired_tests.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写入 {OUT / 'tab_bwt.tex'} 与 reports/bwt_paired_tests.json")


if __name__ == "__main__":
    main()
