# -*- coding: utf-8 -*-
"""从 NCM 汇总 JSON 直接生成论文 LaTeX 表格，避免手抄数字。

用法：
    python -m scripts.make_paper_tables            # 打印三张表并写入 reports/tables/
    python -m scripts.make_paper_tables --check    # 只校验数据齐备性，不写文件

生成三张表：
    tab_main.tex        主表（每基准一列 EWC = 其**调优最优**档 λ=300）
    tab_ewc_lambda.tex  EWC λ 敏感性（四档全列 + 对 FOLoRA 的配对检验）
    tab_ablation.tex    FOLoRA 的 λ / k 消融（含 λ=0 无正交化的对照）

设计约束（改动前先读）：
1. **主表里 EWC 取 λ=300 而不是 §13.3 的判据基线 λ=100**。理由：主表按文献惯例每个
   基线取其调优最优点，而 λ=300 正是 CIFAR 上的最优点（78.00）；把次优点放进主表 =
   欠调最强基线，是 05_experiments.tex 顶部横幅点名的「最该避免的审稿攻击」。
   预注册的判据基线 λ=100 不因此消失——它在 tab_ewc_lambda 里、在正文的 §13.3 判决句里，
   而且**主表题注必须写明全 λ 曲线在 tab_ewc_lambda**，否则主表会被读成「我们赢了」。
2. **不印「n 等长」暗示**。各方法 n 不同（FOLoRA 10/10、Seq 10/5、InfLoRA 5/3、
   CODA 5/3、SimpleCIL 1/1），题注按族逐条列清。
3. INR 无 λ=3000，该格输出 `---` 并**不是**漏跑，是根本没跑；题注要说明。
4. 数字来源 = reports/ncm_summary_{bench}.json（由 **scripts/eval_ncm_sweep.py** 聚合
   落盘；scripts/aggregate.py 是旧的可训练头协议聚合器，与本表无关）。
   配对检验复用 scripts/significance.py 的函数，不另写一份逻辑。
"""

import argparse
import json
from pathlib import Path

from scripts import significance as S

SUMMARY = {
    "cifar100": Path("reports/ncm_summary_cifar100.json"),
    "imagenetr": Path("reports/ncm_summary_imagenetr.json"),
}
OUT = Path("paper/tables")

# ---------------------------------------------------------------------------
# 主表行：(显示名, method, cifar tag, inr tag)
# tag 允许为 None = 该基准上没有这一行（不编造）。
MAIN_ROWS = [
    ("Seq-LoRA", "seq", "default", "default"),
    # EWC 取调优最优点；判据基线 λ=100 见 tab_ewc_lambda
    ("EWC-LoRA", "ewc", "ewc_lam300", "ewc_lam300"),
    ("O-LoRA", "olora", "default", "default"),
    ("L2P", "l2p", "pilot20", "pilot20"),
    ("CODA-Prompt", "coda", "pool100_len8_ep20", "pool100_len8_ep20_inr"),
    ("InfLoRA", "inflora", "default", "default"),
    ("SimpleCIL (floor)", "simplecil", "frozen", "frozen"),
    ("FOLoRA (ours)", "folora_v2", "v2f_l3_k64", "v2f_l3_k64"),
]

# EWC λ 敏感性：(λ, cifar tag, inr tag)
EWC_LAMBDAS = [
    (100, "default", "default"),        # §13.3 判据基线，必须列首位
    (300, "ewc_lam300", "ewc_lam300"),
    (1000, "ewc_lam1000", "ewc_lam1000"),
    (3000, "ewc_lam3000", None),        # INR 未跑 λ=3000
]

OURS = ("folora_v2", "v2f_l3_k64")

# 消融面板：(λ, k, tag)
ABL_LAMBDA = [           # k=64 固定
    (0, 64, "v2f_l0_k64"),       # 去掉 Fisher 加权正交项 = 方法的主要贡献
    (1, 64, "v2f_l1_k64"),
    (3, 64, "v2f_l3_k64"),       # 主配置（§13.3 预注册）
    (10, 64, "v2f_l10_k64"),
]
ABL_K = [                # λ=3 固定（主配置的 λ）
    (3, 16, "v2f_l3_k16"),
    (3, 64, "v2f_l3_k64"),
    (3, 128, "v2f_l3_k128"),
]
# 等权消融（batch `unweighted`）：与 Fisher 加权对照
ABL_EQ = [
    (3, 64, "v2f_eq_l3_k64", "v2f_l3_k64"),
    (10, 64, "v2f_eq_l10_k64", "v2f_l10_k64"),
]


def load_summary(bench: str):
    p = SUMMARY[bench]
    if not p.exists():
        raise SystemExit(f"缺少 {p}（先跑 python -m scripts.aggregate）")
    return json.loads(p.read_text(encoding="utf-8"))


def cell(summ: dict, method: str, tag, key: str, fmt="%.2f$\\pm$%.2f"):
    """取一个 (method, tag) 的 mean±std×100 单元格；缺行返回 None。"""
    if tag is None:
        return None
    rec = summ.get(f"{method}/{tag}")
    if rec is None:
        return None
    return fmt % (rec[f"{key}_mean"] * 100, rec[f"{key}_std"] * 100)


def n_of(summ: dict, method: str, tag):
    if tag is None:
        return None
    rec = summ.get(f"{method}/{tag}")
    return None if rec is None else rec.get("n")


def boldify(vals, best_is_max: bool):
    """在一列中给最优值加 \\textbf。**只在唯一最优时加**——并列最优全部加粗，
    会让读者以为我们有优势；并列时统一不加粗并在题注说明。"""
    nums = []
    for v in vals:
        nums.append(None if v is None else float(v.split("$")[0]))
    ok = [x for x in nums if x is not None]
    if not ok:
        return list(vals)
    target = max(ok) if best_is_max else min(ok)
    if ok.count(target) != 1:
        return list(vals)
    return [v if n is None or n != target else "\\textbf{" + v + "}" for v, n in zip(vals, nums)]


def ewc_vs_ours_facts():
    """逐基准、逐 λ 算出 EWC 与 FOLoRA 的配对比较，并找出 EWC 的 ACC 最优档。

    题注里凡是「哪一档显著 / 是不是 parity / EWC 的最优在哪档」的句子，**全部**从这里取数。
    先前这些句子是写死的字符串，结果主表题注与 λ 表题注都断言
    「the accuracy comparison is parity at every point」，而表 2 自己印着
    **ImageNet-R λ=100：+2.85、p=0.0002（带星）** —— 审稿人对照紧挨着的两处就能抓到
    自相矛盾。写死的结论句必错，故一律改算。
    """
    from scripts.significance import compare, find_key, load_runs_ncm
    facts = {}
    for bench, bname in (("cifar100", "CIFAR-100"), ("imagenetr", "ImageNet-R")):
        summ = load_summary(bench)
        grouped = load_runs_ncm(Path("reports/ncm"), bench)
        ours_key = find_key(grouped, OURS[0], OURS[1])
        if ours_key is None:
            raise SystemExit(f"{bench}: 找不到本文方法 {OURS[1]}")
        rows, accs, tags = [], {}, {}
        for lam, tc, ti in EWC_LAMBDAS:
            tag = tc if bench == "cifar100" else ti
            if tag is None:
                continue
            rec = summ.get(f"ewc/{tag}")
            if rec is None:
                continue
            base_key = find_key(grouped, "ewc", tag)
            if base_key is None:
                raise SystemExit(f"{bench}: 找不到 ewc/{tag}")
            r = compare(grouped, bench, ours_key, base_key, "final_acc_cil", paired=True)
            # 遗忘也要算：题注里关于遗忘方向的句子先前也是**写死的**，而写死的方向是错的
            # —— 它断言「EWC 在 INR λ=300 忘得更少」，实测 FGT 差 = −0.41（p=0.0317）
            # 即 FOLoRA 忘得**更少**，方向相反。凡方向性结论一律现算。
            r_fgt = compare(grouped, bench, ours_key, base_key, "forgetting_cil", paired=True)
            accs[lam] = rec["acc_mean"]
            tags[lam] = tag
            rows.append({"lam": lam, "delta": r["delta"], "p": r["p"], "n": r["n_used"],
                         "fgt_delta": r_fgt["delta"], "fgt_p": r_fgt["p"],
                         "ewc_fgt": rec["fgt_mean"] * 100,
                         "ours_fgt": summ[f"{OURS[0]}/{OURS[1]}"]["fgt_mean"] * 100})
        best_lam = max(accs, key=accs.get) if accs else None
        facts[bench] = {"bname": bname, "rows": rows, "accs": accs,
                        "best_lam": best_lam,
                        "best_tag": tags.get(best_lam)}
    return facts


def parity_sentence(facts):
    """据实描述 EWC 各档与 FOLoRA 的**准确率**比较；有显著点就点名，不含糊其辞。"""
    sig = [(f["bname"], r) for f in facts.values() for r in f["rows"] if r["p"] < 0.05]
    if sig:
        named = ", ".join(
            f"{b} at $\\lambda{{=}}{r['lam']}$ (${r['delta']:+.2f}$, $p={r['p']:.4f}$)"
            for b, r in sig)
        head = (" The accuracy comparison is parity at every point tested \\emph{except} "
                + named + ", where the paired difference reaches significance.")
    else:
        head = (" The accuracy comparison is parity at every point tested, so which method"
                " leads depends only on which operating point of the \\emph{baseline} is"
                " read.")
    for f in facts.values():
        ds = [r["delta"] for r in f["rows"]]
        if ds and min(ds) < 0 < max(ds):
            hi = max(f["rows"], key=lambda r: r["delta"])
            lo = min(f["rows"], key=lambda r: r["delta"])
            head += (f" On {f['bname']} the difference changes sign across the grid"
                     f" (${hi['delta']:+.2f}$ at $\\lambda{{=}}{hi['lam']}$ to"
                     f" ${lo['delta']:+.2f}$ at $\\lambda{{=}}{lo['lam']}$), so the ordering"
                     " of the two methods is not stable across the baseline's own tuned"
                     " points.")
    return head


def forgetting_sentence(facts, short=False):
    """据实描述各档的**遗忘**方向。FGT 低者优，故 fgt_delta<0 = FOLoRA 忘得更少。

    先前写死的版本断言方向「consistently the other way」，与实测不符（INR 反向），
    且把 INR λ=100 的 −1.17（p=0.0004，对我方有利且显著）当成了不利结果。
    """
    against, favour = [], []
    for f in facts.values():
        for r in f["rows"]:
            if r["fgt_p"] >= 0.05:
                continue
            item = f"{f['bname']} $\\lambda{{=}}{r['lam']}$"
            if not short:
                item += (f" ({r['ewc_fgt']:.2f} against {r['ours_fgt']:.2f},"
                         f" $p={r['fgt_p']:.4f}$)")
            (against if r["fgt_delta"] > 0 else favour).append(item)
    if not against and not favour:
        return " On forgetting the two methods are indistinguishable at every point tested."
    parts = []
    if against:
        parts.append("EWC-LoRA forgets \\emph{less} at " + ", ".join(against))
    if favour:
        parts.append("FOLoRA forgets less at " + ", ".join(favour))
    tail = "." if short else " --- neither direction dominates."
    return " The forgetting comparison is mixed: " + "; ".join(parts) + tail


def build_main():
    s = {b: load_summary(b) for b in SUMMARY}
    facts = ewc_vs_ours_facts()

    # EWC 那一行的 tag 改为**数据驱动**：取该基准上 ACC 最高的档（= 每基准各自的调优最优），
    # 而不是写死 λ=300。写死会犯两个错：CIFAR 恰好是 λ=300 所以看不出来，但 INR 的最优是
    # λ=1000（66.33 对 66.32），于是题注「its best tuned operating point」对 INR 不成立。
    rows = []
    for label, m, tc, ti in MAIN_ROWS:
        if m == "ewc":
            tc = facts["cifar100"]["best_tag"]
            ti = facts["imagenetr"]["best_tag"]
        rows.append((label, m, tc, ti))

    acc_c = [cell(s["cifar100"], m, tc, "acc") for _, m, tc, _ in rows]
    fgt_c = [cell(s["cifar100"], m, tc, "fgt") for _, m, tc, _ in rows]
    acc_i = [cell(s["imagenetr"], m, ti, "acc") for _, m, _, ti in rows]
    fgt_i = [cell(s["imagenetr"], m, ti, "fgt") for _, m, _, ti in rows]

    acc_c = boldify(acc_c, True)
    fgt_c = boldify(fgt_c, False)
    acc_i = boldify(acc_i, True)
    fgt_i = boldify(fgt_i, False)

    # n 只进题注（题注里逐方法写明），不单独占一列：加第 6 列会把 390pt 的
    # review 版心撑爆（见文件头约束 2）。n_of 仍被 --check 用于齐备性校验。
    for label, m, tc, ti in rows:
        if not label.startswith("SimpleCIL"):
            assert n_of(s["cifar100"], m, tc) and n_of(s["imagenetr"], m, ti), \
                f"主表行 {label} 在某基准上缺数据（tag 打错？）"

    # EWC 展示档与 parity 结论都由数据生成，不写死。
    lam_clause = ("; ".join(f"$\\lambda{{=}}{f['best_lam']}$ ({f['bname']})"
                            for f in (facts["cifar100"], facts["imagenetr"])))
    ewc_sentence = (" EWC-LoRA is shown at the best of its tuned points on each benchmark --- "
                    + lam_clause
                    + " --- i.e.\\ the strongest configuration of the baseline, with its"
                    " full $\\lambda$ curve, including the pre-registered judging baseline"
                    " $\\lambda{=}100$, in Table~\\ref{tab:ewc-lambda}." + parity_sentence(facts)
                    + forgetting_sentence(facts, short=True)
                    + " Both directions are detailed in Table~\\ref{tab:ewc-lambda};"
                    " the points not named there are not significant.")

    lines = []
    A = lines.append
    A("% 由 scripts/make_paper_tables.py 生成，请勿手改；改数据请改脚本重跑。")
    A("\\begin{table}[tbp]")
    A("\\centering\\small")
    A("\\caption{Class-incremental results under the frozen-feature nearest-class-mean")
    A("protocol (mean $\\pm$ std over seeds; $\\overline{\\mathrm{ACC}}$ higher is better,")
    A("$\\mathrm{FGT}$ lower is better)." + ewc_sentence.replace("&", "\\&") + " Seed budgets are")
    A("\\emph{not} equal and are stated per method: EWC-LoRA and FOLoRA $n{=}10$ on both")
    A("benchmarks; Seq-LoRA and O-LoRA $10$ (CIFAR-100) / $5$ (ImageNet-R); InfLoRA")
    A("$5$/$3$; L2P $5$/$5$; CODA-Prompt $5$/$3$; SimpleCIL is the deterministic")
    A("frozen-feature floor ($n{=}1$, no variance, not tested). Bold marks the best")
    A("value \\emph{among the rows shown here}; where the best is tied no value is")
    A("bolded." + "}")
    A("\\label{tab:main}")
    A("\\setlength{\\tabcolsep}{4pt}")
    A("\\begin{tabular}{@{}lcccc@{}}")
    A("\\toprule")
    A("& \\multicolumn{2}{c}{Split CIFAR-100} & \\multicolumn{2}{c}{Split ImageNet-R} \\\\")
    A("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    A("Method & $\\overline{\\mathrm{ACC}}$ & $\\mathrm{FGT}$ & $\\overline{\\mathrm{ACC}}$ & $\\mathrm{FGT}$ \\\\")
    A("\\midrule")
    for (label, _, _, _), a, f, b, g in zip(rows, acc_c, fgt_c, acc_i, fgt_i):
        if label.startswith("FOLoRA"):
            A("\\midrule")
        A(f"{label} & {a} & {f} & {b} & {g} \\\\")
    A("\\bottomrule")
    A("\\end{tabular}")
    A("\\end{table}")
    return "\n".join(lines)


def build_lambda():
    """EWC λ 敏感性 + 对 FOLoRA 的配对检验（共同 seed）。"""
    from scripts.significance import compare, find_key, load_runs_ncm
    facts = ewc_vs_ours_facts()
    out = ["% 由 scripts/make_paper_tables.py 生成，请勿手改。", "\\begin{table}[tbp]", "\\centering\\small"]
    out.append("\\caption{EWC-LoRA regularization sensitivity. Each $\\lambda$ is compared")
    out.append("against FOLoRA with a paired $t$-test over the seeds common to both;")
    out.append("$n$ is that common count, \\emph{not} either side's own budget, and it is")
    out.append("why the FOLoRA column repeats the same value on every row of a benchmark.")
    out.append("$\\lambda{=}100$ is the baseline fixed in advance, before the seed top-up")
    out.append("that closed the grid to $n{=}10$; $\\lambda{=}3000$ was not run on")
    out.append("ImageNet-R. $^{*}$ marks $p<0.05$." + parity_sentence(facts).replace("&", "\\&"))
    out.append(forgetting_sentence(facts).replace("&", "\\&") + "}")
    out.append("\\label{tab:ewc-lambda}")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    out.append("\\begin{tabular}{@{}lcccccc@{}}")
    out.append("\\toprule")
    out.append("& & & \\multicolumn{2}{c}{$\\overline{\\mathrm{ACC}}$} & & \\\\")
    out.append("\\cmidrule(lr){4-5}")
    out.append("Benchmark & EWC $\\lambda$ & $n$ & EWC & FOLoRA & $\\Delta$ & $p$ \\\\")
    out.append("\\midrule")
    for bench, bname in (("cifar100", "CIFAR-100"), ("imagenetr", "ImageNet-R")):
        summ = load_summary(bench)
        grouped = load_runs_ncm(Path("reports/ncm"), bench)
        ours_key = find_key(grouped, OURS[0], OURS[1])
        if ours_key is None:
            raise SystemExit(f"{bench}: 找不到本文方法 {OURS[1]}")
        ours_all = summ[f"{OURS[0]}/{OURS[1]}"]
        ours_cell = f"{ours_all['acc_mean']*100:.2f}$\\pm${ours_all['acc_std']*100:.2f}"
        for lam, tc, ti in EWC_LAMBDAS:
            tag = tc if bench == "cifar100" else ti
            if bench == "imagenetr" and lam == 100:
                out.append("\\midrule")
            if tag is None:
                # 不编造：该基准确实没跑这一档
                out.append(f"{bname} & {lam} & --- & --- & {ours_cell} & --- & --- \\\\")
                continue
            base_key = find_key(grouped, "ewc", tag)
            if base_key is None:
                raise SystemExit(f"{bench}: 找不到 ewc/{tag}")
            r = compare(grouped, bench, ours_key, base_key, "final_acc_cil", paired=True)
            ewc_rec = summ[f"ewc/{tag}"]
            ewc_cell = f"{ewc_rec['acc_mean']*100:.2f}$\\pm${ewc_rec['acc_std']*100:.2f}"
            star = "$^{*}$" if r["p"] < 0.05 else ""
            out.append(f"{bname} & {lam} & {r['n_used']} & {ewc_cell} & {ours_cell}"
                       f" & {r['delta']:+.2f} & {r['p']:.4f}{star} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(x for x in out if x != "")


def build_ablation():
    from scripts.significance import compare, find_key, load_runs_ncm
    summ = load_summary("cifar100")
    grouped = load_runs_ncm(Path("reports/ncm"), "cifar100")
    ours_key = find_key(grouped, OURS[0], OURS[1])

    def row(panel, lam, k, tag):
        rec = summ.get(f"folora_v2/{tag}")
        if rec is None:
            raise SystemExit(f"缺少 folora_v2/{tag}")
        name = f"$\\lambda{{=}}{lam}$" if panel == "lambda" else f"$k{{=}}{k}$"
        base = (f"{name} & {rec['n']} & {rec['acc_mean']*100:.2f}$\\pm${rec['acc_std']*100:.2f}"
                f" & {rec['fgt_mean']*100:.2f}$\\pm${rec['fgt_std']*100:.2f}")
        if tag == OURS[1]:
            # 主配置自己和自己比 → ttest_rel(a,a) 返回 nan。**不要印 nan**：
            # 这行是参照点，Δ=0 且无检验可言，画成 "---" 才不误导。
            return base + " & \\emph{(main)} & --- \\\\"
        key = find_key(grouped, "folora_v2", tag)
        if key is None:
            return base + " & --- & --- \\\\"
        r = compare(grouped, "cifar100", ours_key, key, "final_acc_cil", paired=True)
        star = "$^{*}$" if r["p"] < 0.05 else ""
        return base + f" & {r['delta']:+.2f} & {r['p']:.4f}{star} \\\\"

    out = ["% 由 scripts/make_paper_tables.py 生成，请勿手改。", "\\begin{table}[tbp]", "\\centering\\small"]
    out.append("\\caption{Ablation on Split CIFAR-100. $\\Delta$ and $p$ compare each setting")
    out.append("with the pre-registered main configuration ($k{=}64$, $\\lambda{=}3$) using a")
    out.append("paired $t$-test over common seeds; $n$ is each row's own seed budget, so")
    out.append("$\\Delta$ is computed on the intersection and need not equal the difference")
    out.append("of the two means shown. $\\lambda{=}0$ removes the Fisher-weighted")
    out.append("orthogonality penalty and changes nothing else.}")
    out.append("\\label{tab:ablation}")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    # 6 列：Setting, n, ACC, FGT, Δ, p。声明与 \multicolumn 的跨度必须都是 6，
    # 少一列报的是 "Extra alignment tab has been changed to \cr"（不是「列太少」这种好懂的错）。
    out.append("\\begin{tabular}{@{}lccccc@{}}")
    out.append("\\toprule")
    out.append("Setting & $n$ & $\\overline{\\mathrm{ACC}}$ & $\\mathrm{FGT}$ & $\\Delta$ vs main & $p$ \\\\")
    out.append("\\midrule")
    out.append("\\multicolumn{6}{@{}l}{\\emph{$\\lambda$ at $k{=}64$}} \\\\")
    for lam, k, tag in ABL_LAMBDA:
        out.append(row("lambda", lam, k, tag))
    out.append("\\midrule")
    out.append("\\multicolumn{6}{@{}l}{\\emph{$k$ at $\\lambda{=}3$}} \\\\")
    for lam, k, tag in ABL_K:
        out.append(row("k", lam, k, tag))
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验齐备性，不写文件")
    args = ap.parse_args()

    tabs = {"tab_main.tex": build_main(),
            "tab_ewc_lambda.tex": build_lambda(),
            "tab_ablation.tex": build_ablation()}

    for name, text in tabs.items():
        print(f"\n===== {name} =====\n{text}")
    if args.check:
        print("\n[--check] 未写文件。")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in tabs.items():
        (OUT / name).write_text(text + "\n", encoding="utf-8")
    print(f"\n已写入 {OUT}/：{', '.join(tabs)}")


if __name__ == "__main__":
    main()
