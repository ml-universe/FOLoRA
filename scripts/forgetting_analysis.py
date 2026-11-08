"""从 `ncm_trace.jsonl` 算**真实时间遗忘**（E2/E2b），并汇总 E5 的严格原型结果。

为什么不能直接用 `eval_ncm` 的 `forgetting_cil`
----------------------------------------------
`reports/ncm/*.json` 那张矩阵是用**终态** θ_T 提一次特征、再逐任务累积类均值得到的：
`acc[t][j]` 的行号是「候选集里有几个任务」，**不是时间**（本仓库 2026-10 已单独记过这个
语义坑）。所以它算出来的「遗忘」其实是候选集扩张带来的精度落差（论文里叫 GAP），
不是「学过之后又忘了」。真正的遗忘必须每个任务边界用**当时的** θ_t 重提特征——
那就是 `ncm_trace_stride=1` 写出来的 `ncm_trace.jsonl`。

两个量的定义（都从 trace 矩阵 A[t][j] 出发，A[t][j] = 学完任务 t 后在任务 j 测试集上的
CIL 准确率，j ≤ t）：

    BWT        = 1/(T-1) · Σ_{j<T-1} ( A[T-1][j] − A[j][j] )        （标准定义，越低越不忘）
    FGT_max    = 1/T     · Σ_j        ( max_{t≥j} A[t][j] − A[T-1][j] ) （是否掉过又回来的惩罚）

BWT 是主口径（与文献可比），FGT_max 一起打印：两者一致说明曲线单调，不一致就说明
中途有回升，单报 BWT 会掩盖掉。**不要只报一个**。

用法
----
  py -3.11 -m scripts.forgetting_analysis --runs seq:default ewc:ncm_trace_l300 \
      folora_v2:v2f_l3_k64 --out reports/forgetting_true_cifar100.json
  py -3.11 -m scripts.forgetting_analysis --preset e2b     # 预设：E2b 的三条
"""

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from peft_cl.utils.seed import set_seed  # noqa: F401  (保持与其它脚本一致的导入面)


def load_trace(run: Path):
    p = run / "ncm_trace.jsonl"
    if not p.exists():
        return None
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[int(r["task_id"])] = r
    return rows


def metrics_from_trace(rows: dict, num_tasks: int):
    """返回 (bwt, fgt_max, acc_final, n_rows, missing)。缺失行按 nan 处理并如实报告。

    `num_tasks` 必须是该 run 的**真实**任务数（从 config.json 读）。这一点不能省：
    训练还在跑时 `ncm_trace.jsonl` 是**逐任务追加**的，半截文件（例如只剩前 16 个
    任务）算出来的 BWT 是「前 16 个任务上的遗忘」，与跑满 20 个任务的值不是同一个量，
    而且它看起来完全正常。2026-11-04 踩过：EWC λ=300 的 seed2 正在重跑，只写了 16 行
    就被算进了 n=3 的均值。
    """
    ks = sorted(rows)
    if len(ks) < 2:
        return dict(error="trace 少于 2 行", n_rows=len(ks),
                    last_task=(max(ks) if ks else None))
    t_last = max(ks)
    if num_tasks and t_last != num_tasks - 1:
        return dict(error=f"trace 只到任务 {t_last}/T-1（{num_tasks} 个任务没跑完）",
                    n_rows=len(ks), last_task=t_last)
    missing = [j for j in range(t_last + 1) if j not in rows]

    # 主对角线必须齐全（A[j][j]），否则 BWT 无定义
    diag_missing = [j for j in range(t_last) if j not in rows]
    if diag_missing:
        return {"error": f"缺少任务 {diag_missing} 的 trace 行，无法算 BWT"}

    bwt_terms, fgt_terms = [], []
    for j in range(t_last):
        a_jj = rows[j]["acc_cil"][j]
        a_tj = rows[t_last]["acc_cil"][j]
        bwt_terms.append(a_tj - a_jj)
        peak = max(rows[t]["acc_cil"][j] for t in ks if t >= j)
        fgt_terms.append(peak - a_tj)

    # AIA（average incremental accuracy, Rebuffi et al. 2017）：每个边界上「模型当时的模样」
    # 在迄今见过的所有类上的平均准确率，再对边界取平均。与 acc_final 的区别是它把整条
    # 轨迹都算进来，因此对「早期学得慢」敏感 —— 这是 BWT 与末态准确率都看不到的一维。
    aia_terms = []
    for t in ks:
        row = rows[t]["acc_cil"][: t + 1]
        aia_terms.append(sum(row) / len(row))

    acc_final = sum(rows[t_last]["acc_cil"][: t_last + 1]) / (t_last + 1)
    return {
        "bwt": sum(bwt_terms) / len(bwt_terms),
        "fgt_max": sum(fgt_terms) / len(fgt_terms),
        "aia": sum(aia_terms) / len(aia_terms),
        "aia_per_boundary": aia_terms,
        "acc_final": acc_final,
        "n_rows": len(ks),
        "last_task": t_last,
        "missing_rows": missing,
    }


def _num_tasks(run: Path):
    """从 run 的 config.json 读任务数；读不到就返回 None（此时只做弱校验）。"""
    p = run / "config.json"
    if not p.exists():
        return None
    try:
        return int(json.load(open(p, encoding="utf-8"))["num_tasks"])
    except Exception:
        return None


def strict_metrics(run: Path):
    """E5：严格序贯原型（原型冻结在 θ_j）。结果由 trainer 写进 results.json。"""
    p = run / "results.json"
    if not p.exists():
        return None
    d = json.load(open(p, encoding="utf-8"))
    if "strict_acc_cil" not in d:
        return None
    ncm = d.get("ncm_acc_cil") or []
    return {
        "strict_acc_cil": d["strict_acc_cil"] * 100,
        "strict_acc_til": d["strict_acc_til"] * 100,
        "strict_per_task_cil": [round(x * 100, 2) for x in d.get("strict_per_task_cil", [])],
        "ncm_acc_mean_at_boundaries": (round(sum(ncm) / len(ncm) * 100, 2) if ncm else None),
    }


# 这些 tag 是**专门为了拿 trace 而重跑的那一份**，与主表同一份配置（同 λ、同 k、同 seed），
# 不是主表 run 本身（主表 run 没开 `ncm_trace_stride`，跑完就没有逐边界行）。
PRESET_E2B = [("seq", "ncm_trace"), ("ewc", "ncm_trace_l1000"),
              ("ewc", "ncm_trace_l300_trace"), ("folora_v2", "ncm_trace_l3_k64")]


def main() -> None:
    ap = argparse.ArgumentParser(description="真实时间遗忘（BWT / 峰值落差）+ E5 严格原型")
    ap.add_argument("--benchmark", default="cifar100")
    ap.add_argument("--runs", nargs="+", default=None, help="形如 method:tag")
    ap.add_argument("--preset", choices=["e2b", "strict"], default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.preset == "e2b":
        pairs = PRESET_E2B
    elif args.preset == "strict":
        pairs = [("seq", "ncm_strict"), ("ewc", "ncm_strict_l300"),
                 ("folora_v2", "ncm_strict_l3_k64")]
    elif args.runs:
        pairs = [tuple(r.split(":", 1)) for r in args.runs]
    else:
        ap.error("要么给 --runs，要么 --preset")

    out = {}
    for method, tag in pairs:
        rec = {"method": method, "tag": tag, "per_seed": []}
        for s in args.seeds:
            run = Path("experiments") / args.benchmark / method / tag / f"seed{s}"
            rows = load_trace(run)
            sm = strict_metrics(run)
            if rows is None and sm is None:
                rec["per_seed"].append({"seed": s, "status": "无 trace / 无 strict"})
                continue
            entry = {"seed": s}
            if rows is not None:
                m = metrics_from_trace(rows, _num_tasks(run))
                if m and "error" not in m:
                    for k in ("bwt", "fgt_max", "aia", "acc_final"):
                        m[k] = round(m[k] * 100, 2)
                    m.pop("aia_per_boundary", None)
                entry["trace"] = m
            if sm is not None:
                entry["strict"] = sm
            rec["per_seed"].append(entry)
        ok = [e["trace"] for e in rec["per_seed"] if e.get("trace") and "error" not in e["trace"]]
        if ok:
            for k in ("bwt", "fgt_max", "aia", "acc_final"):
                vals = [o[k] for o in ok]
                rec[k] = {"mean": round(st.mean(vals), 2),
                          "sd": round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0,
                          "n": len(vals), "seeds": [o for o in vals]}
        sok = [e["strict"] for e in rec["per_seed"] if e.get("strict")]
        if sok:
            rec["strict_mean"] = round(st.mean(x["strict_acc_cil"] for x in sok), 2)
            rec["strict_n"] = len(sok)
        out[f"{method}/{tag}"] = rec

    print(f"{'run':28s} {'BWT':>14s} {'FGT_max':>14s} {'AIA':>8s} {'ACC_last':>10s}  严格原型CIL")
    print("-" * 96)
    for k, r in out.items():
        if "bwt" in r:
            # n 必须打出来：半截 trace 会被上面挡掉，但不同 run 的可用 seed 数仍可能不同，
            # 只看 ±sd 分不清 n=2 还是 n=3
            b = f"{r['bwt']['mean']:+.2f}±{r['bwt']['sd']:.2f} (n={r['bwt']['n']})"
            f = f"{r['fgt_max']['mean']:+.2f}±{r['fgt_max']['sd']:.2f}"
            ai = f"{r['aia']['mean']:.2f}"
            a = f"{r['acc_final']['mean']:.2f}"
        else:
            b = f = ai = a = "  --  "
        s = f"{r['strict_mean']:.2f} (n={r['strict_n']})" if "strict_mean" in r else "--"
        print(f"{k:28s} {b:>14s} {f:>14s} {ai:>8s} {a:>10s}  {s}")
    print("\n注：BWT/FGT_max 均为百分点，负数表示遗忘。两者应同号同量级；"
          "\n    若 FGT_max 明显大于 |BWT|，说明曲线中途掉过又回升，单报 BWT 会掩盖。")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
