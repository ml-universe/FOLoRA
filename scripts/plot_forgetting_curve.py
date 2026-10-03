"""按 NCM 协议画遗忘曲线（论文实验章第 2、3 张图）。

数据来源
--------
`reports/ncm_matrix/<bench>/<method>__<tag>__seed<k>.json`，由
`scripts/eval_ncm_matrix.py`（监督器阶段 14g）落盘。每个文件里有 20x20 的
`acc_cil`，`acc_cil[t][j]` = **学完任务 t 之后**在任务 j 的测试集上的准确率
（在「已见全部类」上做余弦最近类均值分类，见 `scripts/eval_ncm.py:ncm_cil_eval`）。
`j > t` 的位置是 0（未填充，本脚本不使用）。

口径必须与主表一致（这是本脚本唯一的原则）
--------------------------------------------
两个纵轴都直接由 `acc_cil` 定义，且与 `src/peft_cl/metrics/metrics.py` 里主表用的
标量定义**逐项对应**，不是「另算一套」：

  (a) 目前已学任务的平均准确率  mean_{j<=t} acc[t][j]
      —— 这正是 `average_incremental_accuracy` 在时刻 t 的那一项；最终值
      (t = T-1) 就是 `final_average_accuracy`，即主表 ACC 列。

  (b) 已学任务的**平均 CIL–TIL gap**（t>=1）
      mean_{j<=t} ( max_{j<=t'<=t} acc[t'][j] - acc[t][j] )
      —— 逐步版本的历史峰值减当前值（j=t 的项恒为 0），把最后一步 t=T-1 的值取出来
      **就是**主表 GAP 列（两者同为 T 项平均）。

      注意它**不是遗忘**：`acc` 来自 `ncm_cil_eval`，只用终态模型抽一次特征，类均值
      只累加不覆盖，所以该量恒等于 mean_j( a[j][j] - a[T][j] )，即同一特征空间下
      「候选类数从 k 涨到 (t+1)k」造成的精度衰减 = CIL–TIL gap。
      详见 `scripts/make_paper_tables.py` 文件头的 GAP_DEF 段（含验证与反例存档）。

图的诚实性约束（不要为了「好看」去掉任何一条）
----------------------------------------------
1. 不做任何平滑、插值、重采样；画的是逐 step 的原始值。
2. 不截断到「只显示差异」的窄区间：纵轴下限按所有曲线的全局最小值向下取整到 5 的
   倍数再留余量，并把实际用到的区间与全局最小值一并写进落盘 JSON，供题注如实写明
   两点之间的距离（该距离**不保证**恰好等于预留量，因为取整到 5 的倍数会再引入
   ≤5 的额外余量；题注须按 JSON 的实测算，不要硬编码文字）。
   (b) 的遗忘轴从 0 起。
3. 不同方法的 seed 数 n 可能不同（主表本身就分 n=3/5/10 的批次）。脚本会打印并在
   JSON 里记录每个方法的 n；画图**不**暗示这些 n 相等。
4. CIFAR-100 侧额外画一条「重跑噪声带」±1.29 点（围绕 FOLoRA 曲线），并把
   `floor_applies_to` 写清楚：它是同一配置两次独立重跑得到的差值上限
   (77.26 vs 75.97，见 reports/summary_cifar100.md 与 reports/ncm_sweep_cifar100.log)，
   对所有方法都成立，不是 FOLoRA 的误差棒。ImageNet-R 侧没有测过重跑噪声，
   故**不画**（不要把 CIFAR 的数字搬到 INR 上当装饰）。
5. 只有当某个方法是「主表那一行」的 tag 时才画。同一方法出现多个 tag 视为规格错误，
   直接报错退出，不猜、不合并。

用法
----
  python -m scripts.plot_forgetting_curve --benchmark cifar100
  python -m scripts.plot_forgetting_curve                 # 两个基准都画
  python -m scripts.plot_forgetting_curve --self-test     # 造合成数据跑通全链路

自检（--self-test）只写 `reports/ncm_matrix_selftest/` 与 `paper/figures/_selftest/`，
绝不碰真数据目录；合成图**不得**用于论文。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# matplotlib 只用来出图；Agg 后端保证无显示器也能跑（监督器/CI 环境同样适用）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from peft_cl.utils.io import atomic_write_json, read_json_or_none

# 字体：正文用 elsarticle[review] 的 Computer Modern，图里用衬线体 + CM 数学字体，
# 尽量与正文一致；pdf.fonttype=42 把 TrueType 直接嵌进 PDF（默认的 Type 3 字体
# 有些出版社排版流程会拒收）。
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "font.size": 7,
})

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_ROOT = REPO / "reports" / "ncm_matrix"
DEFAULT_OUT_DIR = REPO / "paper" / "figures"

# 主表行顺序（与 05_experiments.tex 的表行、eval_ncm_matrix.py 的 MAIN_TAGS 一致）。
METHOD_ORDER = [
    ("seq", "Seq-LoRA"),
    ("ewc", "EWC-LoRA"),
    ("olora", "O-LoRA"),
    ("l2p", "L2P"),
    ("coda", "CODA-Prompt"),
    ("folora_v2", "FOLoRA"),
]

# Okabe--Ito 色盲友好配色；线型/标记同时变化，黑白打印也能区分。
STYLE = {
    "seq": ("#000000", "o", "-"),
    "ewc": ("#0072B2", "s", "--"),
    "olora": ("#009E73", "^", "-."),
    "l2p": ("#E69F00", "v", ":"),
    "coda": ("#CC79A7", "D", (0, (5, 1, 1, 1))),
    "folora_v2": ("#D55E00", "*", "-"),
}

# 重跑噪声（只对 CIFAR-100 测过）：同一配置、同一 seed、两次独立运行的差值。
RERUN_FLOOR = {"cifar100": 1.29, "imagenetr": None}
RERUN_FLOOR_NOTE = ("77.26 vs 75.97 on Split CIFAR-100, identical configuration and seed; "
                    "see reports/summary_cifar100.md and reports/ncm_sweep_cifar100.log")

BENCH_TITLE = {"cifar100": "Split CIFAR-100", "imagenetr": "Split ImageNet-R"}

FIG_WIDTH_IN = 5.394          # 137mm，即 [review] 模式下的 \\textwidth
FIG_HEIGHT_IN = 2.75


# --------------------------------------------------------------------- 读数据

def load_matrices(matrix_root: Path, bench: str):
    """返回 {method: {"tag": str, "seeds": {seed: np.ndarray}}}，并做一致性检查。"""
    d = matrix_root / bench
    if not d.is_dir():
        return {}, f"{d} 不存在（阶段 14g 还没跑？）"

    found: dict[str, dict] = {}
    for f in sorted(d.glob("*.json")):
        obj = read_json_or_none(f)
        if obj is None:
            print(f"  [跳过] {f.name}：读不出（截断/占用）", file=sys.stderr)
            continue
        method = obj.get("method")
        tag = obj.get("tag")
        seed = obj.get("seed")
        acc = obj.get("acc_cil")
        if method is None or acc is None:
            print(f"  [跳过] {f.name}：缺 method/acc_cil", file=sys.stderr)
            continue
        m = np.asarray(acc, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            print(f"  [跳过] {f.name}：acc_cil 不是方阵 {m.shape}", file=sys.stderr)
            continue

        rec = found.setdefault(method, {"tags": set(), "seeds": {}, "files": []})
        rec["tags"].add(tag)
        rec["seeds"][seed] = m
        rec["files"].append(f.name)

    bad = {m: sorted(r["tags"]) for m, r in found.items() if len(r["tags"]) > 1}
    if bad:
        return {}, ("同一个方法命中了多个 tag，主表行的 tag 不唯一：%s。"
                     "这是规格错误（见 eval_ncm_matrix.py 的 MAIN_TAGS），"
                     "脚本不猜、不合并，请先修数据。" % bad)
    return found, None


def aggregate(seeds: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """把多个 seed 的矩阵逐元素平均。返回 (mean, std, n)。"""
    mats = [seeds[s] for s in sorted(seeds)]
    arr = np.stack(mats, axis=0)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1) if len(mats) > 1 else np.zeros_like(mats[0]), len(mats)


# ------------------------------------------------------------------- 两个纵轴

def accuracy_curve(acc: np.ndarray) -> np.ndarray:
    """(a) 学完 t 之后，已见任务的平均准确率。t 索引 0..T-1。"""
    T = acc.shape[0]
    return np.array([acc[t, : t + 1].mean() for t in range(T)])


def forgetting_curve(acc: np.ndarray) -> np.ndarray:
    """(b) 学完 t 之后，已见任务的平均 CIL–TIL gap。t=0 无前序任务，记为 nan。

    与主表 `src/peft_cl/metrics/metrics.py:forgetting()` **逐项对齐**：在时刻 t 对
    j=0..t 共 **t+1** 项取平均（j=t 的「历史峰值减当前值」恒为 0，正对应 metrics 的
    `range(T)`），峰值取 j..t 上的 max（对应 metrics 的 `range(j, T)`）。因此末步
    t=T-1 的取值**就是**主表 GAP 列。

    （函数名与输出文件名保留 `forgetting_*` 是**故意的**：改名会打穿
    `figures/forgetting_curve_*.pdf` 的引用与 `forgetting_curve_*.json` 的消费方。
    但**呈现层的文字一律不得再写 "forgetting"** —— 该量不是遗忘，见模块 docstring。）

    2026-10-03 修：原先 `drops` 只覆盖 `range(t)`（t 项）而 metrics 用 T 项平均，
    两者相差因子 T/(T-1)（T=20 时 5.26%），末点对不上主表。docstring 曾错误地宣称
    两者相同，掩盖了这个 bug。
    """
    T = acc.shape[0]
    out = np.full(T, np.nan)
    for t in range(1, T):
        drops = [acc[j: t + 1, j].max() - acc[t, j] for j in range(t + 1)]
        out[t] = float(np.mean(drops))
    return out


# ----------------------------------------------------------------------- 画图

def plot_benchmark(bench: str, series: dict, out_dir: Path, floor: float | None,
                   stem_suffix: str = "") -> tuple[Path, Path, dict]:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300, sharex=True,
        gridspec_kw={"wspace": 0.28},
    )

    T = next(iter(series.values()))["acc"].shape[0]
    x = np.arange(1, T + 1)
    order = [m for m, _ in METHOD_ORDER]

    for key, rec in sorted(series.items(), key=lambda kv: order.index(kv[0])):
        label = rec["label"]
        color, marker, ls = STYLE.get(key, ("#555555", "x", "-"))

        # 重跑噪声带只画一次、只画在 (a) 上，且围绕 FOLoRA 曲线；图例里写清它不是
        # FOLoRA 的误差棒（对每个方法都成立，只是画在一条上好读）。
        if floor is not None and key == "folora_v2":
            y = 100.0 * rec["acc_curve"]
            ax_a.fill_between(x, y - floor, y + floor, color=color, alpha=0.16,
                              linewidth=0, zorder=1)

        ax_a.plot(x, 100.0 * rec["acc_curve"], color=color, marker=marker, linestyle=ls,
                  linewidth=1.1, markersize=2.6, label=f"{label} (n={rec['n']})", zorder=3)
        yb = 100.0 * rec["fgt_curve"]
        ax_b.plot(x, yb, color=color, marker=marker, linestyle=ls,
                  linewidth=1.1, markersize=2.6, label=label, zorder=3)

    # 纵轴下限：全局最小值向下取整到 5 的倍数再留余量。不为了让差异显眼而截断——
    # 下限与全局最小值一并写进 JSON，题注须**按实测算**两者的距离，不要硬编码
    # （取整到 5 的倍数会在预留量之外再加 ≤5 的余量，所以实际距离常大于预留量）。
    all_acc = np.concatenate([100.0 * r["acc_curve"] for r in series.values()])
    lo = max(0.0, np.floor((all_acc.min() - 3.0) / 5.0) * 5.0)
    hi = np.ceil((all_acc.max() + 2.0) / 5.0) * 5.0
    ax_a.set_ylim(lo, hi)

    # 上限：原先取整到 2 的倍数、余量 1，实测把 CIFAR 的 9.6 顶到 12、ImageNet-R 的
    # 8.8 顶到 10，图里近两成高度是空的。改成取整到 0.5、余量 0.5（CIFAR→10.5，
    # INR→9.5），刻度仍落在 0/2.5/5/7.5/10 这类可读位置。
    # 题注**没有**写 (b) 的上界（只写了 (a) 的下界），所以收紧上限不需要同步改题注。
    fgts = np.concatenate([100.0 * r["fgt_curve"][1:] for r in series.values()])
    ax_b.set_ylim(0.0, max(1.0, np.ceil((fgts.max() + 0.5) / 0.5) * 0.5))

    ax_a.set_xlabel("tasks learned")
    ax_b.set_xlabel("tasks learned")
    ax_a.set_ylabel("average accuracy on tasks seen so far  [%]")
    ax_b.set_ylabel("mean CIL--TIL gap over tasks seen so far  [%]")
    ax_a.set_title("(a) Accuracy", fontsize=8)
    ax_b.set_title("(b) CIL--TIL gap", fontsize=8)

    for ax in (ax_a, ax_b):
        ax.set_xticks([v for v in (1, 5, 10, 15, 20) if v <= T] or [1, T])
        ax.grid(axis="y", color="#cccccc", linewidth=0.5, alpha=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7, length=2)

    handles, labels = ax_a.get_legend_handles_labels()
    if floor is not None:
        handles.append(Patch(facecolor=STYLE["folora_v2"][0], alpha=0.16, edgecolor="none"))
        labels.append(f"rerun floor $\\pm${floor:g} (all methods)")
    # 0.535 = 两个面板合起来的中点（left=0.085, right=0.985）；用 0.5 会明显偏左。
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.535, 0.995),
               ncol=3, frameon=False, fontsize=6.6, handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5, labelspacing=0.35)

    # 不用 tight_layout：图例放在坐标区外（bbox_to_anchor）时它会误判。显式留白，
    # 再由 savefig(bbox_inches="tight") 兜住图例的悬空部分。
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.155, top=0.795)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"forgetting_curve_{bench}{stem_suffix}"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    for p in (pdf_path, png_path):
        tmp = p.with_name(p.name + ".tmp")
        fig.savefig(tmp, format=p.suffix.lstrip("."), bbox_inches="tight", pad_inches=0.02)
        os.replace(tmp, p)
    plt.close(fig)

    meta = {
        "benchmark": bench,
        "ylabel_accuracy": f"average over j<=t of acc[t][j] ({100:.0f}x scaling in figure)",
        "accuracy_axis_range_in_figure": [float(lo), float(hi)],
        "accuracy_min_plotted": float(all_acc.min()),
        "gap_floor_to_min": float(all_acc.min() - lo),
        "rerun_floor_drawn": floor,
        "rerun_floor_applies_to": "all methods" if floor is not None else None,
    }
    return pdf_path, png_path, meta


# ------------------------------------------------------------------- 合成自检

def _synth_matrix(seed: int, level: float, forgetting: float, T: int = 20) -> np.ndarray:
    """造一个「像那么回事」的 20x20 矩阵：最终水平 level，逐步遗忘 forgetting。"""
    rng = np.random.default_rng(seed)
    acc = np.zeros((T, T))
    for t in range(T):
        for j in range(t + 1):
            age = t - j                      # 学完之后又过了多少个任务
            # 刚学完时高 3 点，之后按 forgetting 线性掉到 level。
            v = level + 3.0 - forgetting * (age / max(1, T - 1))
            acc[t, j] = np.clip(v + rng.normal(0, 0.5), 0, 100) / 100.0
    return acc


def self_test(out_dir: Path, matrix_root: Path) -> int:
    """造合成数据 -> 落盘 -> 跑完整个画图链路。合成图不得用于论文。"""
    root = matrix_root.with_name(matrix_root.name + "_selftest")
    for method, label in METHOD_ORDER:
        level = {"seq": 48.0, "ewc": 76.0, "olora": 68.0, "l2p": 60.0,
                 "coda": 62.0, "folora_v2": 77.5}[method]
        fgt = {"seq": 22.0, "ewc": 7.0, "olora": 9.0, "l2p": 14.0,
               "coda": 12.0, "folora_v2": 6.5}[method]
        for seed in range(4):
            atomic_write_json(root / "cifar100" / f"{method}__synth__seed{seed}.json", {
                "benchmark": "cifar100", "method": method, "tag": "synth", "seed": seed,
                "num_tasks": 20, "acc_cil": _synth_matrix(seed, level, fgt).tolist(),
            })
    print(f"[self-test] 合成矩阵写入 {root}/cifar100/（4 seed x {len(METHOD_ORDER)} 方法）")
    code = run("cifar100", root, out_dir / "_selftest", floor=None,
               json_dir=root / "json", stem_suffix="_SELFTEST")
    print("[self-test] 注意：合成图仅用于验证脚本可跑通，绝不可进论文。")
    return code


# ---------------------------------------------------------------------- 主流程

def run(bench: str, matrix_root: Path, out_dir: Path, floor: float | None,
        json_dir: Path | None = None, stem_suffix: str = "") -> int:
    found, err = load_matrices(matrix_root, bench)
    if err:
        print(f"[{bench}] 无法出图：{err}", file=sys.stderr)
        return 2
    if not found:
        print(f"[{bench}] {matrix_root / bench} 里一个矩阵都没有 —— 阶段 14g 尚未完成，先不出图。",
              file=sys.stderr)
        return 3

    missing = [m for m, _ in METHOD_ORDER if m not in found]
    extra = [m for m in found if m not in [x for x, _ in METHOD_ORDER]]
    if missing:
        print(f"[{bench}] 缺主表方法 {missing}：只有 {sorted(found)}。"
              f"曲线不完整，拒绝出图（用 --allow-missing 可强行出图）。", file=sys.stderr)
        return 4

    series, seeds_report = {}, {}
    print(f"[{bench}] 命中 {sum(len(r['seeds']) for r in found.values())} 个 run：")
    for method, label in METHOD_ORDER:
        rec = found[method]
        mean, std, n = aggregate(rec["seeds"])
        acc_curve = accuracy_curve(mean)
        series[method] = {
            "method": method, "label": label, "n": n, "tag": sorted(rec["tags"])[0],
            "acc": mean, "acc_std": std,
            "acc_curve": acc_curve, "fgt_curve": forgetting_curve(mean),
            "seeds": sorted(rec["seeds"]),
        }
        seeds_report[method] = {"n": n, "seeds": sorted(rec["seeds"]),
                                "tag": sorted(rec["tags"])[0],
                                "final_acc": float(100.0 * acc_curve[-1]),
                                "final_fgt": float(100.0 * forgetting_curve(mean)[-1])}
        print("   %-12s n=%d  tag=%-18s ACC(final)=%6.2f  FGT(final)=%5.2f"
              % (label, n, sorted(rec["tags"])[0], 100 * acc_curve[-1],
                 100 * forgetting_curve(mean)[-1]))

    ns = {r["n"] for r in seeds_report.values()}
    if len(ns) > 1:
        print("   [警告] 各方法的 seed 数 n 不一致，题注必须如实写明，且不可暗示等 n。",
              file=sys.stderr)
    if extra:
        print(f"   [提示] 目录里还有未列为方法名的 run（不画）：{extra}", file=sys.stderr)

    pdf_path, png_path, meta = plot_benchmark(bench, series, out_dir, floor,
                                              stem_suffix=stem_suffix)
    meta.update({
        "matrix_root": str(matrix_root / bench),
        "figure_pdf": str(pdf_path), "figure_png": str(png_path),
        "methods": seeds_report,
        "n_uniform": len(ns) == 1,
        "rerun_floor_value": floor,
        "rerun_floor_note": RERUN_FLOOR_NOTE if floor is not None else None,
        "synthetic": bool(stem_suffix),
        "series": {m: {"tasks": list(range(1, len(r["acc_curve"]) + 1)),
                       "avg_accuracy_seen_so_far": [float(100 * v) for v in r["acc_curve"]],
                       "avg_forgetting_prev_tasks": [None if np.isnan(v) else float(100 * v)
                                                     for v in r["fgt_curve"]]}
                   for m, r in series.items()},
    })
    # 汇总数字落在数据目录的上一级（reports/），与 ncm_matrix 同源，便于对账。
    json_path = (json_dir or matrix_root.parent) / f"forgetting_curve_{bench}{stem_suffix}.json"
    atomic_write_json(json_path, meta)
    print(f"[{bench}] 图 -> {pdf_path}")
    print(f"[{bench}] 数字 -> {json_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="按 NCM 协议画遗忘曲线")
    p.add_argument("--benchmark", default="all", choices=["cifar100", "imagenetr", "all"])
    p.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--no-floor", action="store_true",
                   help="不画重跑噪声带（用于 ImageNet-R，或想出一张纯曲线图时）")
    p.add_argument("--allow-missing", action="store_true",
                   help="缺方法时仍出图（默认拒绝，以免把不完整的曲线当完整结果）")
    p.add_argument("--self-test", action="store_true",
                   help="用合成数据跑通全链路，写 _selftest 目录，绝不污染真数据")
    args = p.parse_args()

    matrix_root = Path(args.matrix_root)
    out_dir = Path(args.out_dir)

    if args.self_test:
        return self_test(out_dir, matrix_root)

    benches = ["cifar100", "imagenetr"] if args.benchmark == "all" else [args.benchmark]
    rc = 0
    for bench in benches:
        floor = None if args.no_floor else RERUN_FLOOR.get(bench)
        code = run(bench, matrix_root, out_dir, floor)
        if code == 4 and args.allow_missing:
            code = run_with_missing(bench, matrix_root, out_dir, floor)
        rc = rc or code
    return rc


def run_with_missing(bench: str, matrix_root: Path, out_dir: Path,
                     floor: float | None) -> int:
    """--allow-missing：把缺失的方法从 METHOD_ORDER 里剔掉再出图。"""
    global METHOD_ORDER
    found, err = load_matrices(matrix_root, bench)
    if err or not found:
        print(f"[{bench}] 仍然无法出图：{err}", file=sys.stderr)
        return 2
    saved = METHOD_ORDER
    try:
        METHOD_ORDER = [(m, l) for m, l in METHOD_ORDER if m in found]
        print(f"[{bench}] --allow-missing：只画 {[l for _, l in METHOD_ORDER]}", file=sys.stderr)
        return run(bench, matrix_root, out_dir, floor)
    finally:
        METHOD_ORDER = saved


if __name__ == "__main__":
    sys.exit(main())
