# -*- coding: utf-8 -*-
"""从 NCM 汇总 JSON 直接生成论文 LaTeX 表格，避免手抄数字。

用法：
    python -m scripts.make_paper_tables            # 打印三张表并写入 paper/tables/
    python -m scripts.make_paper_tables --check    # 只校验数据齐备性，不写文件

生成三张表：
    tab_main.tex        主表（每基准一列 EWC = 该基准的**调优最优**档，数据现算）
    tab_ewc_lambda.tex  EWC λ 敏感性（四档全列 + 对 FOLoRA 的配对检验）
    tab_ablation.tex    FOLoRA 的 λ / k 消融（含 λ=0 无正交化的对照）

设计约束（改动前先读）：
1. **主表里 EWC 取「每个基准各自的调优最优点」，由数据现算，不写死某一档**
   （`build_main` 用 `ewc_vs_ours_facts()[<bench>]["best_tag"]`，即该基准上 ACC 均值
   最高的那一档）。理由：主表按文献惯例每个基线取其调优最优点；写死档位会犯两个错
   ——CIFAR 恰好是 λ=300（78.00）所以看不出来，但 ImageNet-R 的最优是 λ=1000
   （66.33，对 λ=300 的 66.32），若写死 λ=300 则题注「its best tuned operating point」
   对 INR 不成立。把次优点放进主表 = 欠调最强基线，是 05_experiments.tex 顶部横幅
   点名的「最该避免的审稿攻击」。
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
    # O-LoRA / InfLoRA 走 `_fix1`（P0-1：旧 tag 的 checkpoint 只存了末个任务的 adapter，
    # 「评估的模型 ≠ 训练的模型」）。切换点与 significance.py 的 `FIX1` 常量同步，
    # 由 tests/test_fix1_tag_consistency.py 守着，防止只切一半。
    ("O-LoRA", "olora", f"default{S.FIX1}", f"default{S.FIX1}"),
    ("L2P", "l2p", "pilot20", "pilot20"),
    ("CODA-Prompt", "coda", "pool100_len8_ep20", "pool100_len8_ep20_inr"),
    ("InfLoRA", "inflora", f"default{S.FIX1}", f"default{S.FIX1}"),
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
        # 产出这份 JSON 的是 eval_ncm_sweep，不是 aggregate（旧的可训练头协议聚合器）。
        # 按旧的提示去跑 aggregate 会进死胡同：那个脚本根本不写 ncm_summary_*.json。
        raise SystemExit(
            f"缺少 {p}（先跑 python -m scripts.eval_ncm_sweep --benchmark {bench}）")
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


def confirmatory_holm(bench):
    """复算 Table 1 的**确认性家族**（§5 Setup 声明的那一个），返回
    `{(method, tag, metric): row}`，每行带 `p`（原始）与 `p_holm`（校正）。

    为什么在这里重算，而不是读 `reports/significance_*.json`：那份产物的口径
    （`--family` / `--all-configs` / 是否 paired）可以与本次生成的表格不同，
    读文件会让**表里的星号**与**正文声明的家族**悄悄漂移，而这类不一致正是
    本文件顶部注释反复警告的那类 bug（写死的结论句与表体自相矛盾）。
    家族的定义只允许有一处，即 `significance.PAPER_MAIN_NCM`。

    家族 = 同一 benchmark、同一指标下 Table 1 的全部基线对照（cifar m=9、inr m=8）。
    """
    from scripts.significance import compare, find_key, load_runs_ncm
    grouped = load_runs_ncm(Path("reports/ncm"), bench)
    ours_key = find_key(grouped, OURS[0], OURS[1])
    if ours_key is None:
        raise SystemExit(f"{bench}: 找不到本文方法 {OURS[0]}/{OURS[1]}")
    rows = []
    for label, method, tag in S.PAPER_MAIN_NCM[bench]:
        k = find_key(grouped, method, tag)
        if k is None:
            raise SystemExit(f"{bench}: 找不到基线 {method}/{tag}")
        for metric, _ in S.METRICS:
            r = compare(grouped, bench, ours_key, k, metric, paired=True)
            r.update({"baseline_label": label})
            rows.append(r)
    S.apply_holm(rows, "bench-metric")
    return {(r["base_method"], r["base_tag"], r["metric"]): r for r in rows}


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
        # 确认性家族只算一次（下面每一档都从它取数），避免逐档重算 ÷ 家族漂移
        fam = confirmatory_holm(bench)
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
            # 直接从**确认性家族**取，而不是另算一遍：这样 λ 表里的星号与 Table 1
            # 的 p_holm 必然同源。两处各算一次迟早会因为共同 seed 集合或家族口径不同
            # 而给出不同的显著性判定，而两张表在论文里是并排印的。
            # gap 方向也要现算：题注里关于该栏方向的句子先前是**写死的**，而写死的方向是错的
            # —— 它断言「EWC 在 INR λ=300 忘得更少」，实测差 = −0.41（p=0.0317）即
            # FOLoRA 更小，方向相反。凡方向性结论一律现算。
            # （栏名已由 FGT 更正为 GAP，见文件上方 GAP_DEF。数据键仍是缓存的
            #  `fgt_mean`/`forgetting_cil` —— 那是 `eval_ncm_sweep` 的落盘格式，
            #  改它会打穿全部历史缓存，故只改呈现层。）
            r = fam.get(("ewc", tag, "final_acc_cil"))
            r_gap = fam.get(("ewc", tag, "forgetting_cil"))
            if r is None or r_gap is None:
                raise SystemExit(
                    f"{bench}: 确认性家族里没有 ewc/{tag} 的 acc/gap 两行 —— "
                    f"说明 PAPER_MAIN_NCM 与 EWC_LAMBDAS 的 tag 表已经漂移，"
                    f"必须同步二者，不能只改一处")
            accs[lam] = rec["acc_mean"]
            tags[lam] = tag
            rows.append({"lam": lam, "delta": r["delta"], "p": r["p"], "n": r["n_used"],
                         "p_holm": r["p_holm"],
                         "gap_delta": r_gap["delta"], "gap_p": r_gap["p"],
                         "gap_p_holm": r_gap["p_holm"],
                         "ewc_gap": rec["fgt_mean"] * 100,
                         "ours_gap": summ[f"{OURS[0]}/{OURS[1]}"]["fgt_mean"] * 100})
        best_lam = max(accs, key=accs.get) if accs else None
        facts[bench] = {"bname": bname, "rows": rows, "accs": accs,
                        "best_lam": best_lam,
                        "best_tag": tags.get(best_lam)}
    return facts


def parity_sentence(facts, sign_clause: bool = True):
    """据实描述 EWC 各档与 FOLoRA 的**准确率**比较；有显著点就点名，不含糊其辞。

    `sign_clause=False` 时把「逐基准符号翻转」的长句压成一句指向表 2 的提示。
    用途：主表（Table 1）题注**必须**在同一页留下「EWC 的最优档在我们之上、符号随档位
    翻转」的警示，否则主表会被读成「我们赢了」；但完整枚举（每个基准的 hi/lo 档位与
    数值）在 Table 2 的题注里已经印了一遍，主表重复它是纯冗余 —— 而那正是 Table 1
    浮体超高 40.49pt（`main.log` 的 "Float too large for page"）、被推到后页的原因。
    压成一句指针把警示留住、把冗余去掉。**不要**因为排版就整句删掉：删掉警示等于
    把「主表看着像赢」还给审稿人。
    """

    sig = [(f["bname"], r) for f in facts.values() for r in f["rows"]
           if (r["p_holm"] is not None and r["p_holm"] < 0.05)]
    if sig:
        named = ", ".join(
            f"{b} at $\\lambda{{=}}{r['lam']}$ (${r['delta']:+.2f}$, "
            f"$p_{{\\mathrm{{Holm}}}}={r['p_holm']:.4f}$)"
            for b, r in sig)
        head = (" The accuracy comparison is parity at every point tested \\emph{except} "
                + named + ", where the paired difference reaches significance after"
                  " correction.")
    else:
        head = (" The accuracy comparison is parity at every point tested, so which method"
                " leads depends only on which operating point of the \\emph{baseline} is"
                " read.")
    flipped = []
    for f in facts.values():
        ds = [r["delta"] for r in f["rows"]]
        if ds and min(ds) < 0 < max(ds):
            flipped.append(f)
            if not sign_clause:
                continue
            hi = max(f["rows"], key=lambda r: r["delta"])
            lo = min(f["rows"], key=lambda r: r["delta"])
            head += (f" On {f['bname']} the difference changes sign across the grid"
                     f" (${hi['delta']:+.2f}$ at $\\lambda{{=}}{hi['lam']}$ to"
                     f" ${lo['delta']:+.2f}$ at $\\lambda{{=}}{lo['lam']}$), so the ordering"
                     " of the two methods is not stable across the baseline's own tuned"
                     " points.")
    if flipped and not sign_clause:
        names = " and ".join(f["bname"] for f in flipped)
        head += (f" On {names} the difference changes sign across the baseline's own tuned"
                 " points, so the ordering of the two methods is not stable there"
                 " (full curve in Table~\\ref{tab:ewc-lambda}).")
    return head


# ---------------------------------------------------------------------------
# 「遗忘」这一栏的真实语义（2026-10-03 定案，见 内部决策日志）
# ---------------------------------------------------------------------------
# NCM 报告里的 `forgetting_cil` **不是** Chaudhry 遗忘，不能叫 FGT。
# 依据（逐行读源码 + 数值验证，不是推测）：
#   `scripts/eval_ncm.py::ncm_cil_eval` 只用**终态模型抽一次**特征（tr/te 在循环外
#   归一化），类均值 `means[c]` 在 t 增大时只**累加、从不覆盖**。于是 acc[t][j] 随 t
#   变化的唯一原因是**候选类数**从 k 涨到 (t+1)k，与"模型是否忘了"无关。
#   数值恒等式：fgt == mean_j( a[j][j] - a[T][j] )。
#   实测验证：峰值落在 t=j 的比例 = 20/20（逐个方法、两个基准），
#   且 mean_j(max_{t>=j} a[t][j] - a[T][j]) 与 mean_j(a[j][j] - a[T][j]) 差 < 1e-9。
# 故它是 **CIL–TIL gap**：同一特征空间下，把任务 j 的测试集从「只对自家 k 类分类」
# 换成「对全部已见类分类」掉多少分。低者优（更不依赖 task identity）——这正好是
# §5.4 已有的 task-ID-free 叙事，不换故事、只换成正确的名字。
#
# 反例存档（**不要**拿它当遗忘上报）：`results.json` 的 head 协议 forgetting_cil
# 确实是真 Chaudhry 遗忘，但对**所有**方法都 ≈ 84–91（CIFAR 区间仅 88.80–90.90），
# 因为 head 协议在 CIL 下塌到接近随机（final_acc_cil ≈ 7–9%）。无区分度，报上去
# 只会自毁。它仍以 `head_forgetting_cil` 存在缓存里备查，但不进正文。
#
# 亦注意：**不要**用 `final_acc_til - final_acc_cil` 当 gap —— 缓存里的
# `final_acc_til` 来自 `results.json`（head 协议），与 NCM 的 `final_acc_cil`
# 跨协议，相减无意义（实测 O-LoRA seed0：0.8280-0.6674=16.06 ≠ gap 9.26）。
GAP_SYM = "\\mathrm{GAP}"
GAP_DEF = ("$\\mathrm{GAP}$ is the CIL--TIL gap: the mean drop, in points, when a"
           " task's test set is classified against all seen classes rather than its"
           " own, under a single final-model feature extraction (lower is better:"
           " less reliance on task identity).")


def gap_sentence(facts, short=False):
    """据实描述各档的 **CIL–TIL gap** 方向。GAP 低者优，故 gap_delta<0 = FOLoRA 更小。

    注意措辞已从「遗忘」改为「gap」：原句断言「EWC-LoRA 忘得更少」，而该栏量的
    根本不是遗忘（见上）。方向结论本身仍由数据现算，不写死。
    """
    against, favour = [], []
    for f in facts.values():
        for r in f["rows"]:
            # 显著性判据用**校正后**的 p：Setup 已声明本节以 Holm 为准，
            # 若这里仍用原始 p，题注会把正文刚说「不成立」的档位当成显著方向列出来。
            if r["gap_p_holm"] is None or r["gap_p_holm"] >= 0.05:
                continue
            item = f"{f['bname']} $\\lambda{{=}}{r['lam']}$"
            if not short:
                item += (f" ({r['ewc_gap']:.2f} against {r['ours_gap']:.2f},"
                         f" $p_{{\\mathrm{{Holm}}}}={r['gap_p_holm']:.4f}$)")
            (against if r["gap_delta"] > 0 else favour).append(item)
    if not against and not favour:
        return (" On the CIL--TIL gap the two methods are indistinguishable at every"
                " point tested.")
    if short:
        # 主表题注的短版：只报**方向计数**，不逐个枚举档位。
        # 完整枚举（每档的数值与 p）在 Table 2 的题注里已印一遍，主表重复它是纯冗余，
        # 也正是 Table 1 浮体超高被推到后页的原因之一。计数由数据现算，不写死。
        #
        # 计数为 0 的方向**不能照样列出来**：2026-10-04 引入 Holm 校正后，CIFAR-100 侧
        # 已没有任何 gap 档存活，原先的「EWC-LoRA has the smaller gap at 0 of the tuned
        # points tested and FOLoRA at 1」会印出一个计数为 0 的方向。单边/双边分开写。
        n_a, n_f = len(against), len(favour)
        if n_a == 0 or n_f == 0:
            who = "FOLoRA" if n_a == 0 else "EWC-LoRA"
            other = "EWC-LoRA" if n_a == 0 else "FOLoRA"
            n = n_f if n_a == 0 else n_a
            return (f" On the CIL--TIL gap {who} has the \\emph{{smaller}} gap at the"
                    f" {n} tuned point{'s' if n != 1 else ''} where the difference"
                    f" survives correction, and {other} at none; the full curve is in"
                    " Table~\\ref{tab:ewc-lambda}.")
        return (" The CIL--TIL gap comparison is mixed: EWC-LoRA has the \\emph{smaller}"
                f" gap at {n_a} of the tuned points tested and FOLoRA at"
                f" {n_f}; both directions are detailed in"
                " Table~\\ref{tab:ewc-lambda}.")
    parts = []
    if against:
        parts.append("EWC-LoRA has the \\emph{smaller} CIL--TIL gap at "
                     + ", ".join(against))
    if favour:
        parts.append("FOLoRA has the smaller gap at " + ", ".join(favour))
    tail = "." if short else " --- neither direction dominates."
    return " The CIL--TIL gap comparison is mixed: " + "; ".join(parts) + tail


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

    # 第 4 个实参 "fgt" 是 ncm_summary 里的**数据键**（历史命名，来自
    # eval_ncm_sweep 落盘的 fgt_mean/fgt_std）。语义已更正为 GAP，但键名不改：
    # 改键会打穿全部历史缓存。呈现层与数据层的命名不一致是**故意的**。
    acc_c = [cell(s["cifar100"], m, tc, "acc") for _, m, tc, _ in rows]
    gap_c = [cell(s["cifar100"], m, tc, "fgt") for _, m, tc, _ in rows]
    acc_i = [cell(s["imagenetr"], m, ti, "acc") for _, m, _, ti in rows]
    gap_i = [cell(s["imagenetr"], m, ti, "fgt") for _, m, _, ti in rows]

    acc_c = boldify(acc_c, True)
    gap_c = boldify(gap_c, False)
    acc_i = boldify(acc_i, True)
    gap_i = boldify(gap_i, False)

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
                    " $\\lambda{=}100$, in Table~\\ref{tab:ewc-lambda}."
                    + parity_sentence(facts, sign_clause=False)
                    + gap_sentence(facts, short=True))

    lines = []
    A = lines.append
    A("% 由 scripts/make_paper_tables.py 生成，请勿手改；改数据请改脚本重跑。")
    A("\\begin{table}[tbp]")
    A("\\centering\\small")
    A("\\caption{Class-incremental results under the frozen-feature nearest-class-mean")
    A("protocol (mean $\\pm$ std over seeds; $\\overline{\\mathrm{ACC}}$ higher is better).")
    A(GAP_DEF + ewc_sentence.replace("&", "\\&") + " Seed budgets are")
    A("\\emph{not} equal: EWC-LoRA and FOLoRA $n{=}10$ on both benchmarks; Seq-LoRA and")
    A("O-LoRA $10$/$5$ (CIFAR-100/ImageNet-R); InfLoRA $5$/$3$; L2P $5$/$5$; CODA-Prompt")
    A("$5$/$3$; SimpleCIL is the deterministic floor ($n{=}1$, no variance, untested).")
    A("Bold marks the best value \\emph{among the rows shown}; ties are not bolded." + "}")
    A("\\label{tab:main}")
    A("\\setlength{\\tabcolsep}{4pt}")
    A("\\begin{tabular}{@{}lcccc@{}}")
    A("\\toprule")
    A("& \\multicolumn{2}{c}{Split CIFAR-100} & \\multicolumn{2}{c}{Split ImageNet-R} \\\\")
    A("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    A("Method & $\\overline{\\mathrm{ACC}}$ & $\\mathrm{GAP}$ & $\\overline{\\mathrm{ACC}}$ & $\\mathrm{GAP}$ \\\\")
    A("\\midrule")
    for (label, _, _, _), a, f, b, g in zip(rows, acc_c, gap_c, acc_i, gap_i):
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
    out.append("ImageNet-R. The printed $p$ is uncorrected; $^{*}$ marks $p<0.05$ "
               "\\emph{after} Holm--Bonferroni correction within the confirmatory family "
               "of Table~\\ref{tab:main} (Sec.~\\ref{sec:setup}), so a row may print "
               "$p<0.05$ without a star."
               + parity_sentence(facts).replace("&", "\\&"))
    out.append(gap_sentence(facts).replace("&", "\\&") + "}")
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
        fam = confirmatory_holm(bench)
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
            # 从确认性家族取，与 Table 1 的 p_holm 同源（见 confirmatory_holm 注释）
            r = fam.get(("ewc", tag, "final_acc_cil"))
            if r is None:
                raise SystemExit(f"{bench}: 确认性家族里没有 ewc/{tag} 的 ACC 行")
            ewc_rec = summ[f"ewc/{tag}"]
            ewc_cell = f"{ewc_rec['acc_mean']*100:.2f}$\\pm${ewc_rec['acc_std']*100:.2f}"
            star = "$^{*}$" if r["significant_holm"] else ""
            out.append(f"{bname} & {lam} & {r['n_used']} & {ewc_cell} & {ours_cell}"
                       f" & {r['delta']:+.2f} & {r['p']:.4f}{star} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(x for x in out if x != "")


def ablation_holm_family(bench="cifar100"):
    """消融家族 = 同 protocol 下的**全部** config 与主配置互比，m = 家族大小。

    target 的选择与 `scripts/significance.py --all-configs` 逐字一致（同 protocol
    + 排除主配置自己），这样消融表的 $^{*}$ 与正文引用的「corrected $0.011$ across all
    $28$ CIFAR-100 configurations」同源。若两处各算各的，迟早一个按 28 成员、一个按
    本表可见的十几行，给出不同的显著性判定，而它们在论文里是并排印的。

    返回 {(method, tag, metric): row}，每行带 `p` 与 `p_holm`。
    """
    from scripts.significance import compare, find_key, load_runs_ncm
    grouped = load_runs_ncm(Path("reports/ncm"), bench)
    ours_key = find_key(grouped, OURS[0], OURS[1])
    if ours_key is None:
        raise SystemExit(f"{bench}: 找不到本文方法 {OURS[0]}/{OURS[1]}")
    rows = []
    for k in sorted(grouped):
        if k[2] != ours_key[2] or k == ours_key:
            continue
        for metric, _ in S.METRICS:
            r = compare(grouped, bench, ours_key, k, metric, paired=True)
            rows.append(r)
    S.apply_holm(rows, "bench-metric")
    return {(r["base_method"], r["base_tag"], r["metric"]): r for r in rows}


def build_ablation():
    from scripts.significance import compare, find_key, load_runs_ncm
    summ = load_summary("cifar100")
    grouped = load_runs_ncm(Path("reports/ncm"), "cifar100")
    ours_key = find_key(grouped, OURS[0], OURS[1])
    fam = ablation_holm_family("cifar100")

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
        # 从消融家族取，不另算：见 ablation_holm_family 的注释
        r = fam.get(("folora_v2", tag, "final_acc_cil"))
        if r is None:
            return base + " & --- & --- \\\\"
        star = "$^{*}$" if r["significant_holm"] else ""
        return base + f" & {r['delta']:+.2f} & {r['p']:.4f}{star} \\\\"

    out = ["% 由 scripts/make_paper_tables.py 生成，请勿手改。", "\\begin{table}[tbp]", "\\centering\\small"]
    out.append("\\caption{Ablation on Split CIFAR-100. $\\Delta$ and $p$ compare each setting")
    out.append("with the pre-registered main configuration ($k{=}64$, $\\lambda{=}3$) using a")
    out.append("paired $t$-test over common seeds; $n$ is each row's own seed budget, so")
    out.append("$\\Delta$ is computed on the intersection and need not equal the difference")
    out.append("of the two means shown. $\\lambda{=}0$ removes the Fisher-weighted")
    out.append("orthogonality penalty and changes nothing else. $\\mathrm{GAP}$ is the")
    out.append("CIL--TIL gap of Table~\\ref{tab:main}, defined there; lower is better.")
    out.append("The printed $p$ is uncorrected; $^{*}$ marks $p<0.05$ \\emph{after}")
    out.append("Holm--Bonferroni correction over all CIFAR-100 configurations tested")
    out.append("(Sec.~\\ref{sec:setup}), so a row may print $p<0.05$ without a star.}")
    out.append("\\label{tab:ablation}")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    # 6 列：Setting, n, ACC, FGT, Δ, p。声明与 \multicolumn 的跨度必须都是 6，
    # 少一列报的是 "Extra alignment tab has been changed to \cr"（不是「列太少」这种好懂的错）。
    out.append("\\begin{tabular}{@{}lccccc@{}}")
    out.append("\\toprule")
    out.append("Setting & $n$ & $\\overline{\\mathrm{ACC}}$ & $\\mathrm{GAP}$ & $\\Delta$ vs main & $p$ \\\\")
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


# ---------------------------------------------------------------------------
# 附录表：正文引用了、但没有任何表列出的对照（CCF-C 对标评估报告 §5.1 第 6 项）
# ---------------------------------------------------------------------------
# 分两块：
#  (A) **确认性家族** = Table 1 全部基线对照的 avgACC，逐条给出 Δ 与校正 p。
#      Table 1 本身没有 p 列，所以这一整块的 p 值此前只出现在散文里，无法从任何表复核。
#  (B) **探索性面板** = 同 λ 的「Fisher 加权 vs 等权」、k=16 的 λ 扫描、
#      O-LoRA 训练期正交约束两个强度。这些数字同样只在散文里。
#
# (A) 用校正 p；(B) 只给原始 p —— 与 Sec.~\ref{sec:setup} 声明的口径一致
# （确认性家族校正，其余探索性、家族随用随述），题注里写明，不要悄悄混用。
#
# 注意 (B) 的等权对照必须是**同 λ** 配对（λ=3 对 λ=3，λ=10 对 λ=10）。若改成
# 「主配置 vs 等权 λ=10」，数字会从 −0.03 变成 +0.46 —— 那是 `--all-configs` 产物
# 里 `ours_mean` 恒为主配置造成的经典误读（见 reports/significance README 一节）。
def _appendix_exploratory():
    return [
        ("Fisher-weighted against equal-weighted, same $\\lambda$",
         [(f"$\\lambda{{=}}{lam}$", "folora_v2", ours, "folora_v2", base)
          for lam, _, base, ours in ABL_EQ]),
        ("$k{=}16$ protected directions against the $k{=}64$ main configuration",
         [(f"$\\lambda{{=}}{lam}$", OURS[0], OURS[1], "folora_v2", f"v2f_l{lam}_k16")
          for lam in (3, 10, 30, 100, 300, 1000)]),
    ]


def _holm_within_family(ps):
    """给一个探索性家族内的原始 p 列表，返回**同序**的 Holm 校正值。

    规模就是 len(ps)（这里只用 O-LoRA 正交变体那一组的 2 个点）。定义与
    `scripts/significance.py::apply_holm` 一致：按 p 升序排名，第 rank 小者乘 (m - rank)，
    再做单调回填（保证 p_holm 不降）。None / nan 的原样返回 None，且不参与家族。

    为什么放在这里：正文（§5.2、§5.4）引用的是这几个变体在**其 2 成员家族内**校正后的
    p，此前该值只存在于散文，表里查不到。就地复算即可与散文逐位对上，无需改
    significance.py 的家族机制（确认性家族仍然只有 PAPER_MAIN_NCM 一处定义）。
    """
    idx = [i for i, p in enumerate(ps) if p is not None and p == p]
    out = [None] * len(ps)
    m, prev = len(idx), 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: ps[j])):
        v = max(prev, min((m - rank) * ps[i], 1.0))
        out[i] = v
        prev = v
    return out


def _fmt_p(p):
    """p 值格式化。None 与 `nan` **都**要印成 `n/a`。

    为什么单列一个函数：`nan` 在 TeX 里就是三个普通字母，`f"{nan:.4f}"` 会静静印出
    “nan”而**编译不报错**，审稿人看到的是论文正文里一个未定义的记号。这类错误没有任何
    自动检查会拦住它，只有在这里堵。单 seed / 零方差的配对（如 $k{=}16,\\lambda{=}30$）
    正是产生 nan 的来源。
    """
    if p is None or p != p:
        return "n/a"
    return f"{p:.4f}"


# 两张表而不是一张：A 块 17 行 + B 块 11 行 + 3 行分组标题合起来 300pt 出头，
# 超过一整页的可浮动区（elsarticle[review,12pt] 只有 390pt 宽、正文块约 600pt 高），
# LaTeX 报 `Float too large for page by 307.85pt`，表被挤到页边之外而**编译仍然成功**。
# 拆开后每张各自装得下一页，也顺带让 (A) 与 (B) 的口径差异不再共用一条题注。
_APPENDIX_HEAD = ("\\setlength{\\tabcolsep}{4pt}",
                  "\\begin{tabular}{@{}llcccc@{}}",
                  "\\toprule",
                  "Benchmark & Comparison & $n$ & $\\Delta$ & $p$ & $p_{\\mathrm{Holm}}$ \\\\",
                  "\\midrule")
_APPENDIX_TAIL = ("\\bottomrule", "\\end{tabular}", "\\end{table}")


def build_appendix_a():
    """(A) 确认性家族：Table 1 的全部基线对照，逐条给出 Δ 与校正 p。

    这张表**就是**家族本身，所以 p 两侧都给：原始 p 供与旧产物对照，校正 p 供读结论。
    """
    out = ["% 由 scripts/make_paper_tables.py 生成，请勿手改。",
           "\\begin{table}[tbp]", "\\centering\\small"]
    # 题注短：上面 07_appendix.tex 的散文已经交代了 n / Δ / n<2 三条约定，
    # 题注再复述一遍不仅重复，还会把这张 17 行的表顶出一页（实测超 16pt）。
    out.append("\\caption{The confirmatory family of Sec.~\\ref{sec:setup}: FOLoRA against")
    out.append("each baseline of Table~\\ref{tab:main} on average accuracy, with both the")
    out.append("uncorrected $p$ and the Holm--Bonferroni corrected $p_{\\mathrm{Holm}}$ that")
    out.append("the paper's verdicts rest on. Conventions are stated above; the exploratory")
    out.append("comparisons are in Table~\\ref{tab:appendix-signif-b}.}")
    out.append("\\label{tab:appendix-signif-a}")
    out.extend(_APPENDIX_HEAD)
    out.append("\\multicolumn{6}{@{}l}{\\emph{Confirmatory family: Table~\\ref{tab:main},"
               " average accuracy}} \\\\")
    for bench, bname in (("cifar100", "CIFAR-100"), ("imagenetr", "ImageNet-R")):
        fam = confirmatory_holm(bench)
        for label, method, tag in S.PAPER_MAIN_NCM[bench]:
            r = fam.get((method, tag, "final_acc_cil"))
            if r is None:
                raise SystemExit(f"{bench}: 确认性家族缺 {method}/{tag}")
            out.append(f"{bname} & {label} & {r['n_used']} & {r['delta']:+.2f}"
                       f" & {_fmt_p(r['p'])} & {_fmt_p(r['p_holm'])} \\\\")
    out.extend(_APPENDIX_TAIL)
    return "\n".join(out)


def build_appendix_b():
    """(B) 探索性面板：正文引用了、但别处没有表的对照。

    **除 O-LoRA 正交变体那一组外，只给原始 p。** 这些点分属不同家族（同 λ 等权对照、
    k=16 扫描），家族随用随述；把探索性家族并成一个再校正会把它们变成第三种口径，
    与 Sec.~\\ref{sec:setup} 声明的「确认性家族校正、其余不校正」不一致。

    例外（2026-10-05，P1-9）：O-LoRA 训练期正交变体那一组，正文引用的是**该组 2 成员
    家族内**的校正 p（§5.2「within the two-member family those variants form」、
    §5.4），故那一组同时印 $p_{\\mathrm{Holm}}$，并**补上此前完全缺失的 ImageNet-R 两行**
    （正文引用了 +10.59 / +5.74 / p=0.034，表里却没有）。否则正文的 0.0035 / 0.0129 /
    0.034 在表里无处可查，审稿人无法核对。
    """
    from scripts.significance import compare, find_key, load_runs_ncm
    out = ["% 由 scripts/make_paper_tables.py 生成，请勿手改。",
           "\\begin{table}[tbp]", "\\centering\\small"]
    out.append("\\caption{The remaining comparisons the body quotes in prose, which no")
    out.append("other table covers: variants of the main configuration and of one")
    out.append("baseline, on both benchmarks. These are \\emph{exploratory} --- they were")
    out.append("chosen after seeing the results --- so the $p$ given here is uncorrected")
    out.append("and the family each belongs to is stated where its number is used in")
    out.append("Sec.~\\ref{sec:main_results}. The one exception is the O-LoRA")
    out.append("training-time orthogonality block, whose two $\\lambda_1$ points form a")
    out.append("two-member family that the body quotes \\emph{with} its within-family")
    out.append("correction, so its $p_{\\mathrm{Holm}}$ column is filled in. The columns")
    out.append("are those of Table~\\ref{tab:appendix-signif-a}, and the same conventions")
    out.append("apply. No entry here is a re-analysis: the same seeds and the same cached")
    out.append("features are used throughout.}")
    out.append("\\label{tab:appendix-signif-b}")
    out.extend(_APPENDIX_HEAD)
    out.append("\\multicolumn{6}{@{}l}{\\emph{Exploratory (uncorrected $p$)}} \\\\")
    # 缓存每基准只读一次：否则下面十几行每行各读一遍就是十几次对 300+ 个 JSON 的重复解析。
    grouped = {b: load_runs_ncm(Path("reports/ncm"), b)
               for b in ("cifar100", "imagenetr")}
    for group, rows in _appendix_exploratory():
        out.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{~~{group}}}}} \\\\")
        for label, om, ot, bm, bt in rows:
            ok, bk = find_key(grouped["cifar100"], om, ot), find_key(grouped["cifar100"], bm, bt)
            if ok is None or bk is None:
                # 不编造：缺 run 就把该格留空并说明缺哪一侧，而不是印一行看起来正常的数字
                missing = f"{om}/{ot}" if ok is None else f"{bm}/{bt}"
                print(f"[警告] 附录表跳过 {label}：缺少 {missing}")
                out.append(f"CIFAR-100 & {label} & --- & --- & --- & --- \\\\")
                continue
            r = compare(grouped["cifar100"], "cifar100", ok, bk, "final_acc_cil", paired=True)
            out.append(f"CIFAR-100 & {label} & {r['n_used']} & {r['delta']:+.2f}"
                       f" & {_fmt_p(r['p'])} & --- \\\\")

    # ---- O-LoRA 训练期正交变体：两基准，$p_{\mathrm{Holm}}$ 在 2 成员家族内 ----
    # 分组标题用 \multicolumn{6} 是**不受列宽约束**的：它按自然宽度排版，文字比 6 列总宽
    # 长就直接把表顶出 \textwidth，而 LaTeX 只报一行 Overfull hbox、编译照样成功。
    # 2026-10-05 实测：原标题 "O-LoRA with its training-time orthogonality constraint
    # ($p_{\mathrm{Holm}}$ within the two-member family)"（\small 下约 456pt）超出
    # 390pt 的 textwidth，整表报 Overfull 66.71pt。故标题只留最少辨识信息——
    # "training-time" 已在题注与 §5.2 交代，"two-member family" 保留（它是校正口径的锚）。
    out.append("\\multicolumn{6}{@{}l}{\\emph{~~O-LoRA orthogonality strength "
               "($p_{\\mathrm{Holm}}$: two-member family)}} \\\\")
    for bench, bname in (("cifar100", "CIFAR-100"), ("imagenetr", "ImageNet-R")):
        g = grouped[bench]
        recs = []
        for l1 in ("0.1", "1"):
            # compare() 的约定是 delta = 第一参数 − 第二参数；本表全表按「Δ 为正即偏向
            # FOLoRA」排，故第一参数必须是 OURS（FOLoRA）、第二参数是基线。写反不会改
            # p（配对双侧 t 检验对称），但会把 Δ 的符号整体翻转，与 §5.2 的 +7.53/+5.83
            # 对不上——正是不编造数字这条规矩要防的一类。
            ok = find_key(g, *OURS)
            bk = find_key(g, "olora", f"olora_orth_l{l1}{S.FIX1}")
            if ok is None or bk is None:
                missing = f"{OURS[1]}" if ok is None else f"olora/olora_orth_l{l1}{S.FIX1}"
                print(f"[警告] 附录表(B) 跳过 {bname} O-LoRA orth $\\lambda_1$={l1}：缺少 {missing}")
                recs.append((f"$\\lambda_1{{=}}{l1}$", None))
                continue
            recs.append((f"$\\lambda_1{{=}}{l1}$",
                         compare(g, bench, ok, bk, "final_acc_cil", paired=True)))
        holm = _holm_within_family([r["p"] if r else None for _, r in recs])
        for (label, r), ph in zip(recs, holm):
            if r is None:
                out.append(f"{bname} & {label} & --- & --- & --- & --- \\\\")
            else:
                out.append(f"{bname} & {label} & {r['n_used']} & {r['delta']:+.2f}"
                           f" & {_fmt_p(r['p'])} & {_fmt_p(ph)} \\\\")
    out.extend(_APPENDIX_TAIL)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验齐备性，不写文件")
    args = ap.parse_args()

    tabs = {"tab_main.tex": build_main(),
            "tab_ewc_lambda.tex": build_lambda(),
            "tab_ablation.tex": build_ablation(),
            "tab_appendix_signif_a.tex": build_appendix_a(),
            "tab_appendix_signif_b.tex": build_appendix_b()}

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
