"""从已有 run.log 抽取「每任务训练时长」与「任务边界开销」——**零 GPU**，回应 R1 第 2 段。

R1 要的是「同时比较准确率、遗忘、总训练时间、重要性估计代价、峰值显存」的受控对照。
前两项在主表与 E2 里，后三项此前没有系统化。其中**两项不需要重跑**：

* 总训练时间 = Σ(`=== task N ===` → `task N done`)，run.log 每行都有 `HH:MM:SS` 时间戳。

**这个区间包含什么（重要，别记错）**：按 trainer 的调用顺序
`for epoch: 训练` → `_evaluate(全已见任务)` → `ncm_trace` → `after_task(EWC 的 Fisher /
FOLoRA 的参数梯度 + top-k 谱分解)` → 才 return，而 `task N done` 是 run() 在
`_train_task` 返回**之后**才打的。所以「task wall」= 训练 + 评估 + 重要性更新，
而 `task N done` → `=== task N+1 ===` 之间只剩存盘/日志，实测 0–6 s，没有信息量。

⇒ 本脚本**不试图**从日志里拆出纯 Fisher 耗时：拆不出来，硬拆只能靠猜。
纯重要性估计代价由 `profile_resources.py` 直接给 `param_gradients` 计时得到。

⚠ 已知混杂：`ncm_trace_stride>0` 的 run 在边界上还会算 NCM 轨迹，task wall 会被拉长。
这些 run 单独标 `has_ncm_trace: true`，汇总时不要和纯训练 run 混在一起平均。
⚠ 机器休眠（见 当前机器确有休眠史）会把某个任务的时长放大到几小时，
  汇总前按「> 中位数 × 3」剔除整条 run 并报警，不要拿它算均值。

用法
----
  py -3.11 -m scripts.extract_timing                        # 主表五个方法，seeds 0-2
  py -3.11 -m scripts.extract_timing --seeds 0 1 2 3 4      # 扩到 n=5
  py -3.11 -m scripts.extract_timing --out reports/timing_cifar100.json
"""

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) \| (\w+) \| (.*)$")
RE_START = re.compile(r"^=== task (\d+)/(\d+) ===$")
RE_DONE = re.compile(r"^task (\d+) done:")

DEFAULT_RUNS = [
    ("seq", "default"),
    ("ewc", "ewc_lam300"),
    ("olora", "default_fix1"),
    ("inflora", "default"),
    ("folora_v2", "v2f_l3_k64"),
]


def _hm2s(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_log(path: Path) -> dict:
    """返回每任务的训练段秒数、边界段秒数，以及跨天校正后的总时长。

    时间戳只有 `HH:MM:SS`、没有日期，所以跨午夜时必须自己补日：只要后一行比前一行
    「早」超过 12 小时，就认定跨了一天（真实相邻行的间隔不可能接近半天）。
    """
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        mt = TS.match(line)
        if mt:
            rows.append((_hm2s(mt.group(1), mt.group(2), mt.group(3)),
                         mt.group(5).strip()))

    # 跨午夜校正：逐行累加 day 偏移
    abs_ts, day, prev = [], 0, None
    for t, msg in rows:
        if prev is not None and t < prev - 12 * 3600:
            day += 1
        abs_ts.append(t + day * 86400)
        prev = t

    train_s, boundary_s = [], []
    cur_start = None
    last_done = None
    for t, msg in zip(abs_ts, [m for _, m in rows]):
        m1 = RE_START.match(msg)
        if m1:
            if last_done is not None:
                boundary_s.append(t - last_done)      # 上一任务结束后到本任务开训
                last_done = None
            cur_start = t
            continue
        m2 = RE_DONE.match(msg)
        if m2 and cur_start is not None:
            train_s.append(t - cur_start)
            cur_start = None
            last_done = t

    return {
        "train_s_per_task": train_s,
        "boundary_s_per_task": boundary_s,
        "train_s_total": sum(train_s),
        "boundary_s_total": sum(boundary_s),
    }


def summarize(vals):
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 2),
            "sd": round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def main() -> None:
    p = argparse.ArgumentParser(description="从 run.log 抽训练/边界时长（零 GPU）")
    p.add_argument("--benchmark", default="cifar100")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--runs", nargs="+", default=None,
                   help="覆盖默认行，格式 method:tag。默认 = 主表五个方法")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    pairs = ([tuple(r.split(":", 1)) for r in args.runs] if args.runs
             else DEFAULT_RUNS)

    out = {}
    for method, tag in pairs:
        agg = {"task_wall_s": [], "runs": [], "excluded": []}
        for s in args.seeds:
            d = Path("experiments") / args.benchmark / method / tag / f"seed{s}"
            log = d / "run.log"
            if not log.exists():
                agg["runs"].append({"seed": s, "status": "no log"})
                continue
            info = parse_log(log)
            has_trace = False
            cfgp = d / "config.json"
            if cfgp.exists():
                cfg = json.load(open(cfgp, encoding="utf-8"))
                has_trace = bool(cfg.get("ncm_trace_stride") or
                                 cfg.get("ncm_strict_proto"))

            walls = info["train_s_per_task"]
            med = statistics.median(walls) if walls else 0.0
            sleep_hit = bool(walls) and max(walls) > 3 * med

            reason = None
            if has_trace:
                reason = "ncm_trace/pronto 打开，task wall 被 NCM 计算拉长"
            elif sleep_hit:
                reason = (f"疑似机器休眠：最长任务 {max(walls):.0f}s > 3×中位数 "
                          f"{med:.0f}s，整条 run 剔除")
            elif not walls:
                reason = "日志里没有完整的 task 段"

            if reason is None:
                agg["task_wall_s"] += walls
            else:
                agg["excluded"].append({"seed": s, "reason": reason})

            agg["runs"].append({
                "seed": s, "task_wall_s_total": info["train_s_total"],
                "inter_task_s_total": info["boundary_s_total"],
                "n_tasks_seen": len(walls),
                "has_ncm_trace": has_trace, "sleep_artifact": sleep_hit,
                "excluded_reason": reason,
            })
        out[f"{method}/{tag}"] = {
            "task_wall_s": summarize(agg["task_wall_s"]),
            "run_total_s": summarize(
                [r["task_wall_s_total"] for r in agg["runs"]
                 if "task_wall_s_total" in r and not r.get("excluded_reason")]),
            "excluded_seeds": agg["excluded"],
            "runs": agg["runs"],
        }

    hdr = f"{'方法':22s} {'task wall/task':>18s} {'整run总时长':>16s}  剔除"
    print(hdr)
    print("-" * len(hdr))
    for k, v in out.items():
        tr, tot = v["task_wall_s"], v["run_total_s"]
        if tr:
            print(f"{k:22s} {tr['mean']:8.1f}s ±{tr['sd']:6.1f} "
                  f"{tot['mean'] / 3600:9.2f}h ±{tot['sd'] / 3600:5.2f} "
                  f"(n={tr['n']})   {len(v['excluded_seeds'])} 条")
        else:
            print(f"{k:22s} 无可用 run（全部剔除或有 NCM 轨迹）")
        for e in v["excluded_seeds"]:
            print(f"    - seed{e['seed']}: {e['reason']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
