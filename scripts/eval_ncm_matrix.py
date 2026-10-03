"""把 NCM 评估的**完整准确率矩阵**落盘（零训练），供画遗忘曲线用。

为什么需要单独一个脚本
----------------------
`scripts/eval_ncm.py` 的 `ncm_cil_eval` **已经算出** 20×20 的准确率矩阵 `acc[t][j]`
（任务 t 结束时在任务 j 上的准确率），并以 `"acc_cil"` 返回；但
`scripts/eval_ncm_sweep.py` 落盘时**只写标量**（ACC/FGT/INC），矩阵被丢掉了。
于是 `reports/ncm/*.json` 里没有矩阵，画不出「按 NCM 协议」的遗忘曲线。

而 `experiments/.../results.json` 里的 `acc_cil` 是**旧的可训练线性头协议**的矩阵，
与主表的 NCM 口径**不一致**——拿它画图会造成「图与表口径不同」，正是要避免的事。

本脚本只做一件事：对**主表用到的那些 run** 重跑一次 NCM 评估，把矩阵存到
`reports/ncm_matrix/<bench>/<method>__<tag>__seed<k>.json`，**不改动任何已有文件**。

自带的正确性检查（重要）
------------------------
每个 run 都把新算出的三个标量与 `reports/ncm/` 里已有的标量逐项比对，把最大绝对
偏差写进结果的 `consistency` 字段并打印。**若不一致，说明 NCM 评估本身不可复现**，
那是比「缺矩阵」严重得多的问题，必须停下来查，不得继续累积数据。

用法
----
  python -m scripts.eval_ncm_matrix --benchmark cifar100
  python -m scripts.eval_ncm_matrix --benchmark imagenetr --max_seeds 5
"""

import argparse
import json
from pathlib import Path

import torch

from peft_cl.utils.io import atomic_write_json, read_json_or_none
from scripts.eval_ncm import evaluate_one
from scripts.eval_ncm_sweep import discover_runs

# 主表实际报的那一行所用的 tag（与 05_experiments.tex 的表行一一对应）。
# 只取主表行：画图要的是「表里那几个方法」的曲线，不是全部 28 个 tag。
MAIN_TAGS = {
    "cifar100": {
        "seq":       ["default"],
        "ewc":       ["ewc_lam300"],          # 预注册规则 3 触发后的主表 EWC 行
        "olora":     ["default"],
        "l2p":       ["pilot20"],             # prompt 方法各自最优（20 epoch）
        "coda":      ["pool100_len8_ep20"],
        "folora_v2": ["v2f_l3_k64"],
    },
    "imagenetr": {
        "seq":       ["default"],
        # 2026-10-03 修：INR 的主表 EWC 行是 λ=1000（66.33，见 tab:main 的说明
        # 「best of its tuned points ... λ=1000 (ImageNet-R)」），而这里原先写
        # ewc_lam300 → 图上标着 "EWC-LoRA" 的画的是 λ=300（65.92），与表不同配置。
        "ewc":       ["ewc_lam1000"],
        "olora":     ["default"],
        "l2p":       ["pilot20"],
        "coda":      ["pool100_len8_ep20_inr"],   # INR 侧的忠实 CODA 配置
        "folora_v2": ["v2f_l3_k64"],
    },
}

SCALARS = ("final_acc_cil", "forgetting_cil", "incremental_acc_cil")


def main():
    p = argparse.ArgumentParser(description="NCM 准确率矩阵落盘（零训练，供画遗忘曲线）")
    p.add_argument("--benchmark", default="cifar100", choices=["cifar100", "imagenetr"])
    p.add_argument("--exp_root", default="experiments")
    p.add_argument("--ncm_root", default="reports/ncm")
    p.add_argument("--out_root", default="reports/ncm_matrix")
    p.add_argument("--max_seeds", type=int, default=5,
                   help="每个 (method,tag) 最多取前多少个 seed（曲线按 seed 均值±std 画，"
                        "5 个足够；调大只是增加 GPU 时间）")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_root = Path(args.exp_root)
    ncm_root, out_root = Path(args.ncm_root), Path(args.out_root)
    tags_map = MAIN_TAGS[args.benchmark]

    runs = discover_runs(exp_root, args.benchmark, list(tags_map))
    picked = []
    per_tag = {}
    for method, tag, seed, seed_dir, _d in runs:
        if tag not in tags_map.get(method, []):
            continue
        k = (method, tag)
        if per_tag.get(k, 0) >= args.max_seeds:
            continue
        per_tag[k] = per_tag.get(k, 0) + 1
        picked.append((method, tag, seed, seed_dir))

    print(f"[{args.benchmark}] 主表 run 命中 {len(picked)} 个（上限 {args.max_seeds} seed/行）",
          flush=True)
    if not picked:
        print("没有命中任何 run —— 检查 MAIN_TAGS 与 experiments/ 的目录名是否一致")
        return

    worst = 0.0
    n_done = n_skip = 0
    for method, tag, seed, seed_dir in sorted(picked):
        out = out_root / args.benchmark / f"{method}__{tag}__seed{seed}.json"
        if out.exists():
            n_skip += 1
            continue
        print(f"  [run] {method}/{tag}/seed{seed}", flush=True)
        res = evaluate_one(args.benchmark, 20, seed, device, run_dir=str(seed_dir),
                           batch_size=args.batch_size, num_workers=args.num_workers)

        ref = read_json_or_none(ncm_root / args.benchmark / out.name)
        consistency = {}
        if ref is not None:
            for k in SCALARS:
                if k in ref and k in res:
                    consistency[k] = abs(float(ref[k]) - float(res[k]))
            if consistency:
                m = max(consistency.values())
                worst = max(worst, m)
                flag = "  <-- 不一致!" if m > 1e-6 else ""
                print("        一致性 max|Δ| = %.3e%s" % (m, flag), flush=True)

        atomic_write_json(out, {
            "benchmark": args.benchmark, "method": method, "tag": tag, "seed": seed,
            "num_tasks": len(res["acc_cil"]),
            "acc_cil": res["acc_cil"],
            "final_acc_cil": res["final_acc_cil"],
            "forgetting_cil": res["forgetting_cil"],
            "incremental_acc_cil": res["incremental_acc_cil"],
            "consistency_vs_ncm": consistency,
        })
        n_done += 1

    print(f"\n完成 {n_done} 个，跳过（已存在）{n_skip} 个 -> {out_root}/{args.benchmark}")
    print(f"与 reports/ncm 标量的最大绝对偏差 = {worst:.3e}")
    if worst > 1e-6:
        print("!! 警告：NCM 评估不可复现（偏差 > 1e-6）。**停止累积数据，先查根因**。")
    else:
        print("一致性检查通过（完全一致）。")


if __name__ == "__main__":
    main()
