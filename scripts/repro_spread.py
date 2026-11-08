"""同配置重复 run 的离散度（E4）——「单次实验的噪声有多大」。

为什么要单独报
--------------
论文里所有「差距」都要和**噪声**比。同一个 seed、同一份配置、同一台机器重跑，
结果照样会变（`reports/folora-nondeterministic-training`：本仓库的训练从 task 0
epoch 0 就分叉，`torch.equal` 类判据一律作废）。所以必须给出「单次 run 的波动」这个
标尺，否则「+0.14 点」这种差异读者无从判断是小还是无所谓。

本脚本读 `experiments/*/*/repro_*/seed*/` 这批重复 run 的 **NCM 缓存**
（`reports/ncm/`，与主表同口径），按组输出 ACC / GAP 的极差与样本标准差。

`--refs` 把主表里**先于本批就存在**的那一条同配置 run 并进来（默认给
`ewc:default,ewc:ewc_lam100,folora_v2:v2f_l3_k64`）。理由是正文原来用
`ewc/default` 与 `ewc/ewc_lam100` 在 seed 0 上的 77.26 vs 75.97（差 1.29 点）当地板，
那是**一对**观测；本批补的 5+3 条与它们是同一条配置（`ewc_lambda` 都是 100；
`default` 唯一不同的字段 `folora_lambda` 经源码确认 EWC 不读），所以应当并成同一样本，
而不是另立一套。`--floor_out` 会据此写出合并地板 `reports/repro_floor.json`：
每臂 sd、方差比检验、合并 sd、以及**两次独立 run 之差的 sd**（= √2·合并 sd，
这才是画在曲线图上的那条带）。

这两个数还被 `scripts/tost_equivalence.py` 用来定等价边界 δ：实测单次波动约 0.5–0.7
点，取 δ=1.0 作为「可忽略」的定义。

用法
----
  py -3.11 -m scripts.repro_spread
  py -3.11 -m scripts.repro_spread --floor_out reports/repro_floor.json
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

METRICS = [("final_acc_cil", "ACC"), ("forgetting_cil", "GAP")]

# 正文原来那对 1.29 点的两端：都是 EWC、λ=100、seed 0，只差 tag。
# `default` 的 folora_lambda=1.0 与其余几条的 300.0 不同，但 EWC 不读这个字段
# （src/peft_cl/methods/ewc.py 里只出现 ewc_lambda），所以算同一条配置。
REFS_DEFAULT = "ewc:default,ewc:ewc_lam100,folora_v2:v2f_l3_k64"


def _pool_floor(arms: dict) -> dict:
    """两臂 sd → 方差比检验 → 合并 sd → 两次独立 run 之差的 sd。

    预注册的报告规则（`run_lambda_grid.py::REPRO_FLOOR_CONFIGS`）：两臂 sd 并列报出；
    量级一致就报合并地板并说明理由，不一致就按较大者取。这里 2 臂时做 F 检验，
    p 明显时**不合并**、返回较大者，并在 JSON 里写明用的是哪条规则。
    """
    from scipy import stats

    names = [m for m, a in arms.items() if a["n"] >= 2]
    if len(names) != 2:
        return {"error": f"需要恰好 2 个 n>=2 的臂，实际 {names}"}
    a, b = arms[names[0]], arms[names[1]]
    va, vb = a["sd"] ** 2, b["sd"] ** 2
    F = max(va, vb) / min(va, vb)
    dfn, dfd = (a["n"] - 1, b["n"] - 1) if va >= vb else (b["n"] - 1, a["n"] - 1)
    p = float(2 * min(stats.f.cdf(F, dfn, dfd), 1 - stats.f.cdf(F, dfn, dfd)))
    pooled = (((a["n"] - 1) * va + (b["n"] - 1) * vb)
              / (a["n"] + b["n"] - 2)) ** 0.5
    merge = p >= 0.05
    sd = pooled if merge else max(a["sd"], b["sd"])
    return {
        "arms": {m: {"n": arms[m]["n"], "mean": round(arms[m]["mean"], 3),
                     "sd": round(arms[m]["sd"], 3),
                     "range": round(arms[m]["range"], 3),
                     "values": arms[m]["values"]} for m in names},
        "variance_ratio": {"F": round(F, 3), "df": [dfn, dfd], "p": round(p, 3)},
        "merged": merge,
        "rule": ("两臂量级一致（p>=0.05）→ 合并" if merge
                 else "两臂量级不一致（p<0.05）→ 取较大者"),
        "single_run_sd": round(sd, 3),
        "n_runs_total": a["n"] + b["n"],
        "pairwise_diff_sd": round(sd * 2 ** 0.5, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="重复 run 的离散度（E4 噪声标尺）")
    ap.add_argument("--cache_root", default="reports/ncm")
    ap.add_argument("--benchmark", default="cifar100")
    ap.add_argument("--prefix", default="repro_",
                    help="只统计 tag 以该前缀开头的 run（默认 repro_）")
    ap.add_argument("--refs", default=REFS_DEFAULT,
                    help="逗号分隔的 method:tag，作为同配置的既有成员并入（'' 关闭）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--floor_out", default=None,
                    help="写出合并地板 JSON（每臂 sd / F 检验 / 差值 sd）")
    args = ap.parse_args()

    refs = {}
    for item in (args.refs or "").split(","):
        item = item.strip()
        if not item:
            continue
        m, t = item.split(":", 1)
        refs[(m, t)] = True

    groups = defaultdict(list)
    base = Path(args.cache_root) / args.benchmark
    for p in sorted(base.glob("*.json")):
        rec = json.load(open(p, encoding="utf-8"))
        is_repro = str(rec.get("tag", "")).startswith(args.prefix)
        # **必须卡 seed==0**：`default` / `v2f_l3_k64` 这些 tag 在 seed 0–9 上各有一条
        # 缓存，只按 tag 匹配会把 10 个种子的 run 全吸进来，把「同种子重跑」变成
        # 「跨种子波动」（2026-11-04 踩过：EWC 臂一度算出 n=16、sd=0.94）。
        is_ref = (rec["method"], rec["tag"]) in refs and int(rec.get("seed", -1)) == 0
        if not (is_repro or is_ref):
            continue
        groups[rec["method"]].append(rec)
    if refs:
        # 别静默少一条：正文引用的那条 run 若缓存换了名字，必须炸出来
        for m, t in refs:
            if not any(r["tag"] == t for r in groups.get(m, [])):
                raise SystemExit(f"--refs 里的 {m}:{t} 在 {base} 下找不到 NCM 缓存")

    out = {}
    for method, recs in sorted(groups.items()):
        entry = {"n_runs": len(recs), "tags": [r["tag"] for r in recs]}
        for metric, label in METRICS:
            vals = [r[metric] * 100 for r in recs]
            entry[label] = {
                "values": [round(v, 3) for v in vals],
                "mean": round(st.mean(vals), 3),
                "sd_sample": round(st.stdev(vals), 3) if len(vals) > 1 else 0.0,
                "range": round(max(vals) - min(vals), 3),
            }
        out[method] = entry

    print(f"=== {args.benchmark}: 同配置重复 run 的离散度（n_runs ≥ 2 才有意义）===")
    print(f"{'method':10s} {'n':>3s} {'ACC mean±sd':>18s} {'ACC range':>10s} "
          f"{'GAP mean±sd':>18s} {'GAP range':>10s}")
    print("-" * 74)
    for m, e in out.items():
        a, g = e["ACC"], e["GAP"]
        flag = "" if e["n_runs"] >= 3 else "  ← run 太少，sd 不可靠"
        print(f"{m:10s} {e['n_runs']:3d} "
              f"{a['mean']:9.2f} ±{a['sd_sample']:5.2f} {a['range']:10.2f} "
              f"{g['mean']:9.2f} ±{g['sd_sample']:5.2f} {g['range']:10.2f}{flag}")

    sds = [out[m]["ACC"]["sd_sample"] for m in out if out[m]["n_runs"] >= 3]
    if sds:
        print(f"\n单次 run 的 ACC 标准差约 {min(sds):.2f}–{max(sds):.2f} 点；"
              f"TOST 的 δ=1.0 取的是它的约两倍。")

    floor = None
    if args.floor_out:
        arms = {m: {"n": e["n_runs"], "mean": e["ACC"]["mean"],
                    "sd": e["ACC"]["sd_sample"], "range": e["ACC"]["range"],
                    "values": e["ACC"]["values"]}
                for m, e in out.items() if e["n_runs"] >= 3}
        floor = _pool_floor(arms)
        if "error" not in floor:
            print(f"\n=== 合并地板 ===\n{floor['rule']}")
            for m, a in floor["arms"].items():
                print(f"  {m:12s} n={a['n']} mean={a['mean']:.2f} sd={a['sd']:.2f} "
                      f"range={a['range']:.2f}")
            print(f"  方差比 F({floor['variance_ratio']['df'][0]},"
                  f"{floor['variance_ratio']['df'][1]})={floor['variance_ratio']['F']:.2f} "
                  f"p={floor['variance_ratio']['p']:.3f}")
            print(f"  单次 run sd = {floor['single_run_sd']:.2f}（n={floor['n_runs_total']}）；"
                  f"**两次独立 run 之差的 sd = {floor['pairwise_diff_sd']:.2f}** ← 曲线图的带")
            # 旧口径：正文曾把 `ewc/default` 与 `ewc/ewc_lam100` 这一对当 floor
            by_tag = {r["tag"]: r["final_acc_cil"] * 100
                      for r in groups.get("ewc", [])}
            if "default" in by_tag and "ewc_lam100" in by_tag:
                g = by_tag["default"] - by_tag["ewc_lam100"]
                floor["legacy_pair"] = {
                    "tags": ["default", "ewc_lam100"],
                    "values": [round(by_tag["default"], 3),
                               round(by_tag["ewc_lam100"], 3)],
                    "gap": round(g, 3),
                    "in_single_run_sd": round(abs(g) / floor["single_run_sd"], 2),
                    "in_pairwise_diff_sd": round(abs(g) / floor["pairwise_diff_sd"], 2),
                }
                print(f"  旧正文那一对 {by_tag['default']:.2f} vs {by_tag['ewc_lam100']:.2f}"
                      f" = {g:.2f} 点（= {abs(g) / floor['single_run_sd']:.1f} 个单次 sd）"
                      f" —— 是极值对，不是 sd")
            ewc_vals = floor["arms"].get("ewc", {}).get("values") or []
            if ewc_vals:
                g = max(ewc_vals) - min(ewc_vals)
                floor["ewc_range"] = {"gap": round(g, 3), "n": len(ewc_vals),
                                      "in_single_run_sd": round(g / floor["single_run_sd"], 2)}
                print(f"  EWC 臂 {len(ewc_vals)} 条的极差 = {g:.2f} 点"
                      f"（= {g / floor['single_run_sd']:.1f} 个单次 sd）")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"写入 {args.out}")
    if args.floor_out and floor and "error" not in floor:
        Path(args.floor_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(floor, open(args.floor_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"写入 {args.floor_out}")


if __name__ == "__main__":
    main()
