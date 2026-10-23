"""汇总界的实证探针（`bound_terms.jsonl`），并按**预先冻结**的阈值给出结论。

为什么需要单独的汇总脚本
------------------------
`src/peft_cl/utils/bound_probe.py` 在**每个任务边界**写一行记录（T=20 → 最多 19 行），
一行只反映一个边界的 (θ_{t-1}, θ_t, F̄_{<t}) 三元组。论文的两条承诺说的是整体性质，
所以必须先跨边界聚合成一个数，再判定。

**不要**用 `scripts/bound_terms.py` 汇总：它从 checkpoint 事后重算，而 `after_task`
已把 `ref_params` 覆盖成当前参数，实测逐层 ‖cur-ref‖ 全为 0.000000 —— 它得到的
dL(α) 恒为 0、比值为 nan，把 nan 读成「三阶项为零」是错的。已在该脚本顶部标红。

冻结的判定规则（2026-09-24 先于数据写下，数据到了不改阈值）
----------------------------------------------------------
每个可用边界 t 计算  r_t = |c_t| / |b_t|   （α=1 处「三阶项 / 二阶项」的量级比）。
以**中位数** median(r) 判定 04_theory.tex 的承诺 1「高阶项相对二阶项可忽略」：

  * median(r) <= 0.10  → KEEP     ：承诺原样成立
  * 0.10 < median(r) <= 0.30 → WEAKEN：改为「与二阶项同量级但不主导」
  * median(r) > 0.30   → DELETE   ：删掉承诺句（不改写成弱版本，直接不承诺）

阈值 0.10 / 0.30 不是新发明的：`scripts/bound_terms.py` 的单 run 判定本来就用这两个
数，这里只是把它们从「单个 checkpoint」搬到「跨边界中位数」。

**另**：无论中位数落在哪一档，都要同时报出 max(r) 与 r>0.30 的边界个数。
若中位数达标而个别边界严重超标，正文必须写出最坏情形——只用中位数报喜就是选择性汇报。

承诺 2（Assumption 1 的违背量级）没有通过/不通过，只做**量化**：
  q_t = |a_t| / |b_t|，即线性残留在 α=1 处相对二阶项的占比。
中位数与最大值都要写进 Discussion。若中位数 > 1，说明该工作点上 Assumption 1 被实质
违背、界的「主项」不再主导，**必须显著披露**，不能只报 fisher_over_fitted 好看的那部分。

用法
----
  python -m scripts.bound_probe_report                       # 扫 experiments/cifar100 下全部 jsonl
  python -m scripts.bound_probe_report --root experiments/cifar100/folora_v2/probe_l10_k16
  python -m scripts.bound_probe_report --min_tasks 5         # 少于 5 个可用边界的 run 不参与判定
"""

import argparse
import json
import statistics as st
from pathlib import Path

KEEP_MAX = 0.10
WEAKEN_MAX = 0.30


def load_records(root: Path):
    """读 root 下所有 bound_terms.jsonl，返回 [(run_dir, [rec, ...]), ...]。"""
    out = []
    for p in sorted(root.glob("**/bound_terms.jsonl")):
        recs = []
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                # 探针写入被中断时最后一行可能是半截 JSON。丢弃并报出，不静默。
                print(f"  [warn] {p}: 第 {ln} 行无法解析（{e}）；已丢弃")
        if recs:
            out.append((p.parent, recs))
    return out


def _ratio(name_a: str, name_b: str, rec: dict):
    """取 |a|/|b|；b 为 0 或缺失、非有限值时返回 None（不把 nan 当数据）。"""
    a, b = rec.get(name_a), rec.get(name_b)
    if a is None or b is None:
        return None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if b == 0.0 or not (a == a) or not (b == b):     # nan 安全检查
        return None
    if b != b or a in (float("inf"), float("-inf")):
        return None
    return abs(a) / abs(b)


def summarise(run_dir: Path, recs: list) -> dict:
    r_cubic, r_lin, fisher_ratio, deltas = [], [], [], []
    for rec in recs:
        rc = _ratio("fit_c_cubic", "fit_b_quadratic", rec)
        rl = _ratio("fit_a_linear", "fit_b_quadratic", rec)
        if rc is not None:
            r_cubic.append(rc)
        if rl is not None:
            r_lin.append(rl)
        if rec.get("delta_norm") is not None:
            deltas.append(float(rec["delta_norm"]))
        ff = rec.get("fisher_over_fitted")
        if isinstance(ff, (int, float)) and ff == ff and ff not in (float("inf"), float("-inf")):
            fisher_ratio.append(float(ff))

    n_bad = sum(1 for r in r_cubic if r > WEAKEN_MAX)
    return {
        "run": str(run_dir),
        "n_boundaries_total": len(recs),
        "n_boundaries_usable": len(r_cubic),
        "task_ids": sorted(int(r["task_id"]) for r in recs if "task_id" in r),
        "r_cubic_median": st.median(r_cubic) if r_cubic else None,
        "r_cubic_max": max(r_cubic) if r_cubic else None,
        "r_cubic_min": min(r_cubic) if r_cubic else None,
        "n_boundaries_over_weaken_max": n_bad,
        "r_linear_median": st.median(r_lin) if r_lin else None,
        "r_linear_max": max(r_lin) if r_lin else None,
        "fisher_over_fitted_median": st.median(fisher_ratio) if fisher_ratio else None,
        "delta_norm_median": st.median(deltas) if deltas else None,
        "per_boundary_r_cubic": r_cubic,
    }


def verdict(median_r):
    if median_r is None:
        return "NO-DATA", "没有任何可用边界（|b|=0 或记录缺失）——不能用它支撑任何承诺"
    if median_r <= KEEP_MAX:
        return "KEEP", ("承诺 1 原样成立：「the higher-order term is negligible "
                        "relative to the quadratic term」有实测支撑")
    if median_r <= WEAKEN_MAX:
        return "WEAKEN", ("三阶项与二阶项同量级但不主导 → 正文改为"
                          "「of the same order as, but not dominant over, the quadratic term」"
                          "，不得再写 negligible")
    return "DELETE", ("三阶项主导/与二阶项相当 → **删掉承诺句**（04_theory.tex 的那句 "
                      "'We verify empirically that ...'），直接不承诺，不要改写成弱版本")


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="汇总界的实证探针并给冻结规则下的结论")
    ap.add_argument("--root", default="experiments/cifar100",
                    help="搜索 bound_terms.jsonl 的根目录")
    ap.add_argument("--min_tasks", type=int, default=3,
                    help="可用边界少于此数的 run 仅展示、不参与判定")
    ap.add_argument("--out", default="reports/bound_probe_report.json")
    args = ap.parse_args()

    found = load_records(Path(args.root))
    if not found:
        print(f"在 {args.root} 下没找到任何 bound_terms.jsonl。")
        print("阶段 3i（supervise_all）是否已跑过？见 reports/supervise_all.log。")
        print("注意：探针默认关闭，只有 --bound_probe 的 run 才会产出该文件。")
        return 1

    summaries = []
    for run_dir, recs in found:
        s = summarise(run_dir, recs)
        summaries.append(s)
        print(f"\n=== {run_dir} ===")
        print(f"  边界记录 {s['n_boundaries_total']} 条，其中可比 {s['n_boundaries_usable']} 条"
              f"（task_id {s['task_ids'][:3]}…{s['task_ids'][-1:] if s['task_ids'] else []}）")
        if s["delta_norm_median"] is not None:
            print(f"  δθ 范数中位数 = {s['delta_norm_median']:.4f}")
        if s["r_cubic_median"] is None:
            print("  无可用的 |c|/|b| —— 所有边界的 b 都为 0，无法判定")
            continue
        print(f"  |c|/|b|  中位 {s['r_cubic_median']:.4f}   "
              f"最大 {s['r_cubic_max']:.4f}  最小 {s['r_cubic_min']:.4f}   "
              f"(> {WEAKEN_MAX} 的边界数: {s['n_boundaries_over_weaken_max']})")
        print(f"  |a|/|b|  中位 {s['r_linear_median']:.4f}  最大 {s['r_linear_max']:.4f}")
        if s["fisher_over_fitted_median"] is not None:
            print(f"  (½δθᵀF̄δθ)/(拟合二阶) 中位 = {s['fisher_over_fitted_median']:.3f}"
                  f"   —— 远离 1 说明 Fisher 估计未标定实测曲率")

    judged = [s for s in summaries if s["n_boundaries_usable"] >= args.min_tasks
              and s["r_cubic_median"] is not None]
    print("\n" + "=" * 72)
    if not judged:
        print("没有 run 满足 --min_tasks，无法给出结论。")
        print("（单 run 的探测只有 1 个 seed；边界数不足时不要把单边界数字当结论用。）")
    else:
        all_r = [r for s in judged for r in s["per_boundary_r_cubic"]]
        med = st.median(all_r)
        v, why = verdict(med)
        print(f"跨 run 汇总：可用边界共 {len(all_r)} 条，|c|/|b| 中位数 = {med:.4f}")
        print(f"结论 = {v}")
        print(f"  {why}")
        worst = max(all_r)
        print(f"最坏边界 |c|/|b| = {worst:.4f}"
              + ("  ← 中位数达标但最坏情形严重，正文必须写出最坏情形"
                 if worst > WEAKEN_MAX >= med else ""))
        lins = [s["r_linear_median"] for s in judged if s["r_linear_median"] is not None]
        if lins:
            lin_med = st.median(lins)
            print(f"承诺 2（Assumption 1 的违背量级）：|a|/|b| 中位数 = {lin_med:.4f}")
            print("  → 这个数要写进 Sec. discussion；> 1 说明该工作点上主项不再主导，"
                  "必须显著披露")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"frozen_rule": {"keep_max": KEEP_MAX, "weaken_max": WEAKEN_MAX},
                    "summaries": summaries},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
