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
  python -m scripts.eval_ncm_matrix --benchmark imagenetr --max_seeds 10
"""

import argparse
import json
from pathlib import Path

import torch

from peft_cl.utils.io import atomic_write_json, read_json_or_none
from scripts.eval_ncm import evaluate_one, source_fingerprint
from scripts.eval_ncm_sweep import MATRIX_MAX_SEEDS, discover_runs
from scripts.significance import FIX1

# 主表实际报的那一行所用的 tag（与 05_experiments.tex 的表行一一对应）。
# 只取主表行：画图要的是「表里那几个方法」的曲线，不是全部 28 个 tag。
#
# O-LoRA 走 `default{FIX1}` 而不是写死 "default"：P0-1 的旧 checkpoint 只存了末个任务的
# adapter，画在 Fig 2/3 上的会是「评估的模型 ≠ 训练的模型」。后缀统一从
# scripts/significance.py 的常量取，**不要就地写死**——切一半的表现是图和表用了不同
# 一批 run，两者看起来都正常。见 tests/test_fix1_tag_consistency.py。
MAIN_TAGS = {
    "cifar100": {
        "seq":       ["default"],
        "ewc":       ["ewc_lam300"],          # 预注册规则 3 触发后的主表 EWC 行
        "olora":     [f"default{FIX1}"],
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
        "olora":     [f"default{FIX1}"],
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
    # 默认取 10 以与主表 seed 预算一致（P1-1）：主表用 n=10，而此前的默认 5 会让
    # Fig 2/3 的末点最多比主表差 0.55 点，造成图-表口径不一致。
    p.add_argument("--max_seeds", type=int, default=MATRIX_MAX_SEEDS,
                   help="每个 (method,tag) 最多取前多少个 seed（曲线按 seed 均值±std 画；"
                        "默认 10，与主表 seed 预算一致）")
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
            # 内容指纹比对（P2-5，同 eval_ncm_sweep）：索引含 tag，正常换 tag 重训即换
            # 键；此处防的是**同 tag 原地覆盖**——那时不比对就会沿用旧实现的数字。
            # 旧缓存无 src_fp 则按「无法判断」沿用，避免作废既有缓存。
            prev = read_json_or_none(out)
            src_fp = source_fingerprint(seed_dir)
            old_fp = None if prev is None else prev.get("src_fp")
            if src_fp is not None and old_fp is not None and old_fp != src_fp:
                print(f"  [stale] {method}/{tag}/seed{seed}: checkpoint 已变"
                      f"（缓存 size={old_fp.get('size')} -> 实为 {src_fp['size']}），重算",
                      flush=True)
            else:
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

        rec = {
            "benchmark": args.benchmark, "method": method, "tag": tag, "seed": seed,
            "src_fp": source_fingerprint(seed_dir),   # 实际评估的那份 checkpoint
            "num_tasks": len(res["acc_cil"]),
            "acc_cil": res["acc_cil"],
            "final_acc_cil": res["final_acc_cil"],
            "forgetting_cil": res["forgetting_cil"],
            "incremental_acc_cil": res["incremental_acc_cil"],
            "consistency_vs_ncm": consistency,
        }
        # 与 eval_ncm_sweep / eval_ncm_matrix_extra 对齐：把权重加载自检一并落盘。
        # 此前本脚本**逐键列举**写出，而 evaluate_one 后来新增的 load_audit 不在列表里，
        # 于是同一份评估在 reports/ncm/ 有审计、在 reports/ncm_matrix/ 没有 ——
        # 而画论文 Fig 2/3 读的正是后者。**此写法改为「先建 dict 再补键」，
        # 以后 evaluate_one 再加字段时这里不会静默漏掉。**
        if "load_audit" in res:
            rec["load_audit"] = res["load_audit"]
        atomic_write_json(out, rec)
        n_done += 1

    print(f"\n完成 {n_done} 个，跳过（已存在）{n_skip} 个 -> {out_root}/{args.benchmark}")
    print(f"与 reports/ncm 标量的最大绝对偏差 = {worst:.3e}")
    if worst > 1e-6:
        print("!! 警告：NCM 评估不可复现（偏差 > 1e-6）。**停止累积数据，先查根因**。")
    else:
        print("一致性检查通过（完全一致）。")


if __name__ == "__main__":
    main()
