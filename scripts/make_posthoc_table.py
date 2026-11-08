# -*- coding: utf-8 -*-
"""生成 `paper/tables/tab_posthoc.tex`：**2×2 决策规则 × 主干**的分解表（R8-4）。

这张表回答的是审稿人 R8-4 的问题：我们表里的排序，有多少来自「表示」，有多少
来自「最近类均值」这条决策规则？做法是拿**同一个 checkpoint**换成 RanPAC
（NeurIPS 2023，训练无关的闭式分类器）再评一次，看排序会不会翻。

数据来源与守门
--------------
* RanPAC 列 = `reports/ranpac_*.json`（由 `scripts/eval_ranpac.py` 落盘）。
  只接受 `M=10000`、`rp_seed=1993` 的正式配置；`M=0`（消融）与
  `rp_seed=1994/1995`（随机投影种子敏感性）的文件**不进本表**，否则同一列里
  混着不同配置，读表的人无从知道自己在看什么。
* NCM 列 = `reports/ncm/{bench}/*.json` **同一 seed** 的那一条，保证两列来自
  同一个模型。**绝不用 `ncm_summary_*.json` 的 10-seed 均值**：那是主表的口径，
  与本表的单 seed 混在一行会差 0.4~0.5 分，读者会以为是「换头带来的差」。
* 特征是否真的来自那个 checkpoint，由 `eval_ranpac.py` 里的 `verify_ncm` 对账
  闸门保证；本脚本额外检查 `ncm_verify.ok`，为 `False` 直接拒绝出表。

用法
----
    py -3.11 -m scripts.make_posthoc_table           # 打印并写文件
    py -3.11 -m scripts.make_posthoc_table --check   # 只校验，不写
"""

import argparse
import glob
import json
import statistics as st
from pathlib import Path

OUT = Path("paper/tables")

# 与 `scripts/make_paper_tables.py::MAIN_ROWS` 逐条对齐（tag 一个字都不能差）。
# 顺序即表的行序：把「冻结」放第一行，读者才能一眼看出后面每一行相对它的净增益。
# 第 3/4 个元素是 (cifar_tag, inr_tag)：**EWC 的 λ 逐基准不同**（CIFAR λ=300、
# INR λ=1000，由 make_paper_tables 的 `best_tag` 现算）。两个基准共用同一个 tag
# 会把 INR 那一格换成欠调的 checkpoint，与主表不是同一行。
ROWS = [
    ("Frozen ViT-B/16 (no adaptation)", None, (None, None)),
    ("Seq-LoRA", "seq", ("default", "default")),
    ("EWC-LoRA (tuned)", "ewc", ("ewc_lam300", "ewc_lam1000")),
    ("O-LoRA", "olora", ("default_fix1", "default_fix1")),
    ("InfLoRA", "inflora", ("default_fix1", "default_fix1")),
    ("FOLoRA (ours)", "folora_v2", ("v2f_l3_k64", "v2f_l3_k64")),
]
BENCHES = [("cifar100", "CIFAR-100"), ("imagenetr", "ImageNet-R")]
M_MAIN, RP_MAIN = 10000, 1993


def _key(name: str):
    """`runs` 的键：冻结是 `frozen`，其余是 `method/tag/seedN`。"""
    if name == "frozen":
        return None, None, None
    parts = name.split("/")
    if len(parts) != 3 or not parts[2].startswith("seed"):
        return None, None, None
    return parts[0], parts[1], int(parts[2].replace("seed", ""))


def load_ranpac():
    """返回 {(bench, method|None, tag|None): {seed: final_acc_cil*100}}。"""
    table = {}
    for p in sorted(glob.glob("reports/ranpac_*.json")):
        if "_smoke" in p:
            continue
        d = json.load(open(p, encoding="utf-8"))
        if d.get("M") != M_MAIN or d.get("rp_seed") != RP_MAIN:
            continue
        bench = d.get("benchmark")
        # `class_order_seed` 是 2026-11-04 之后才写进 JSON 的；更早的文件只能从
        # 文件名 `..._seedN.json` 推，推不出来就不猜（宁可缺格）
        fname_seed = None
        stem = Path(p).stem
        for suf in ("_seed", "_s"):
            if stem.endswith(tuple(f"{suf}{i}" for i in range(10))):
                fname_seed = int(stem[-1])
                break
        for name, r in d.get("runs", {}).items():
            if name == "run":
                # `--run_dir` 单跑模式：键名是笼统的 "run"，真正的身份在 `target`
                # 路径里（experiments/<bench>/<method>/<tag>/seed<N>）。**不能**退回
                # 文件名推 seed —— 单跑文件不带方法名，会全部并进冻结主干那一格，
                # 把冻结的均值污染成混合值（2026-11-04 踩过：InfLoRA 的 82.34 被
                # 平均进 frozen 的 81.57/80.91 里，表上还会显示 "---"）。
                tgt = Path(r["target"])
                method, tag = tgt.parent.parent.name, tgt.parent.name
                seed = int(tgt.name.replace("seed", ""))
            else:
                method, tag, seed = _key(name)
                if seed is None:
                    seed = d.get("class_order_seed", fname_seed)
            if seed is None:
                raise SystemExit(f"{p}:{name} 拿不到 class_order 的 seed，拒绝猜")
            if r.get("ncm_verify", {}) and r["ncm_verify"].get("ok") is False:
                raise SystemExit(
                    f"{p}:{name} 的 NCM 对账失败 —— 特征不是这个 checkpoint 的，"
                    "整张表不可用")
            slot = table.setdefault((bench, method, tag), {})
            val = r["final_acc_cil"] * 100
            if seed in slot and abs(slot[seed] - val) > 1e-6:
                # 同一 (基准, 方法, tag, seed) 出现两个不同的值，说明有文件不是同一配置
                raise SystemExit(
                    f"{p}: {bench}/{method}/{tag}/seed{seed} 与另一份文件的数不一致"
                    f"（{slot[seed]:.2f} vs {val:.2f}），拒绝出表")
            slot[seed] = val
    return table


def load_ncm(bench):
    """{(method, tag, seed): final_acc_cil*100}，只取 `reports/ncm/{bench}/`。"""
    out = {}
    for p in glob.glob(f"reports/ncm/{bench}/*.json"):
        d = json.load(open(p, encoding="utf-8"))
        out[(d.get("method"), d.get("tag"), int(d.get("seed", -1)))] = \
            d["final_acc_cil"] * 100
    return out


def _cell(vals):
    if not vals:
        return "---"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{st.mean(vals):.2f}$_{{\\pm {st.pstdev(vals):.2f}}}$"


def paired_tests(rp):
    """FOLoRA 对每个对照主干，在**同一 seed** 上做配对 t 检验（RanPAC 列）。

    只做有 ≥2 个共同 seed 的对照；n=1 时返回 None 而不是给一个假的 p 值。
    不跨 seed 相减：本仓库同 seed 同配置的两次独立 run 之差，sd 约 0.89 点
    （`reports/repro_floor.json`），不配对会把显著性吹大。
    """
    from scipy import stats

    out = {}
    for bench, _ in BENCHES:
        ours = rp.get((bench, "folora_v2", "v2f_l3_k64"), {})
        if len(ours) < 2:
            continue
        # 冻结主干也在内：它是最关键的一条对照（「适配器到底买到了什么」），
        # 而且它有三个 class order，正好与 FOLoRA 的三个 seed 配对
        for method, tags in [(None, (None, None)),
                             ("seq", ("default", "default")),
                             ("ewc", ("ewc_lam300", "ewc_lam1000")),
                             ("olora", ("default_fix1", "default_fix1")),
                             ("inflora", ("default_fix1", "default_fix1"))]:
            tag = tags[0] if bench == "cifar100" else tags[1]
            other = rp.get((bench, method, tag), {})
            seeds = sorted(set(ours) & set(other))
            name = f"{bench}/{'frozen' if method is None else method}"
            if len(seeds) < 2:
                out[name] = {"n": len(seeds), "skipped": "共同 seed < 2"}
                continue
            a = [ours[s] for s in seeds]
            b = [other[s] for s in seeds]
            t, p = stats.ttest_rel(a, b)
            d = [x - y for x, y in zip(a, b)]
            out[name] = {
                "n": len(seeds), "delta": round(sum(d) / len(d), 2),
                "sd_of_diff": round((sum((x - sum(d) / len(d)) ** 2 for x in d)
                                     / (len(d) - 1)) ** 0.5, 2),
                "t": round(float(t), 2), "p": round(float(p), 4), "seeds": seeds}
    return out


def build():
    rp = load_ranpac()
    ncm = {b: load_ncm(b) for b, _ in BENCHES}

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Decision rule $\times$ representation. Each row re-evaluates the "
        r"\emph{same} features with two rules: cosine nearest-class-mean (NCM, the "
        r"protocol used throughout the paper) and RanPAC (a training-free ridge "
        r"classifier with random projections, NeurIPS 2023) fitted on the same "
        r"features. No model is retrained for the RanPAC columns.}",
        r"\label{tab:posthoc}",
        r"\begin{tabular}{l" + "cc" * len(BENCHES) + "}",
        r"\toprule",
        r" & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{nm}}}" for _, nm in BENCHES) + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Feature extractor & " + " & ".join(["NCM & RanPAC"] * len(BENCHES))
        + r" \\",
        r"\midrule",
    ]
    counts = {b: [] for b, _ in BENCHES}  # 每行在该基准上实际有几个 class order
    for label, method, tags in ROWS:
        cells = [label]
        for bi, (bench, _) in enumerate(BENCHES):
            tag = tags[bi]
            if method is None:
                n = ncm[bench].get(("simplecil", "frozen", 0))
                r = rp.get((bench, None, None), {})
            else:
                n = ncm[bench].get((method, tag, 0))
                r = rp.get((bench, method, tag), {})
            cells.append("---" if n is None else f"{n:.2f}")
            cells.append(_cell(sorted(r.values())))
            counts[bench].append(len(r))
        lines.append(" & ".join(cells) + r" \\")

    # 脚注里的「几个 class order」必须从**实际数据**数出来，不能写死。
    # 2026-11-04 踩过：INR 补到三个种子后脚注还写着「单 class order」，
    # 而表里已经带了 $\pm$，自相矛盾。哪一行少跑一个种子，这里就说得出来。
    spans = {b: (min(counts[b]), max(counts[b])) for b, _ in BENCHES}
    if all(s == (3, 3) for s in spans.values()):
        seed_txt = (r"The RanPAC column is the mean$\pm$std over the three class "
                    r"orders (seeds 0--2) on both benchmarks. ")
    else:
        parts = []
        for b, nm in BENCHES:
            lo, hi = spans[b]
            if (lo, hi) == (3, 3):
                parts.append(rf"three class orders (seeds 0--2) on {nm}")
            elif lo == hi:
                parts.append(rf"{lo} class order{'s' if lo > 1 else ''} on {nm}")
            else:
                parts.append(rf"{lo}--{hi} class orders across rows on {nm}")
        seed_txt = (r"The RanPAC column is the mean$\pm$std over "
                    + " and ".join(parts) + ". ")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        # 1pt 而不是 2pt：这张表带长题注正好占满一页，2pt 时 LaTeX 报
        # "Float too large for page by 0.28pt"（肉眼不可见，但会让「零告警」失效）
        r"\par\vspace{1pt}",
        r"\footnotesize",
        r"Both columns use the same terminal checkpoint per row; the NCM column is "
        r"seed~0, so it can differ by $\leq 0.5$ points from the 10-seed means in "
        r"Table~\ref{tab:main}. "
        + seed_txt +
        r"$M{=}10{,}000$ random projections, $\lambda$ selected per task on a held-out "
        r"split of the current task. EWC uses the $\lambda$ that is best tuned for each "
        r"benchmark ($\lambda{=}300$ on CIFAR-100, $\lambda{=}1000$ on ImageNet-R), "
        r"matching Table~\ref{tab:main}. FOLoRA vs.\ the frozen backbone under the RanPAC "
        r"rule isolates what the adapter buys once the decision rule is no longer the "
        r"bottleneck.",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    print(text)
    tests = paired_tests(load_ranpac())
    print("\n配对检验（RanPAC 列，FOLoRA − 对照，同一 class_order seed）：")
    for k, v in tests.items():
        if "skipped" in v:
            print(f"  {k:18s} 跳过：{v['skipped']}（n={v['n']}）")
        else:
            print(f"  {k:18s} Δ={v['delta']:+.2f}±{v['sd_of_diff']:.2f} "
                  f"t={v['t']:+.2f} p={v['p']:.4f} n={v['n']}")
    if args.check:
        print("\n[--check] 未写文件")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_posthoc.tex").write_text(text + "\n", encoding="utf-8")
    json.dump(tests, open("reports/posthoc_paired_tests.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写入 {OUT / 'tab_posthoc.tex'} 与 reports/posthoc_paired_tests.json")


if __name__ == "__main__":
    main()
