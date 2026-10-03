# -*- coding: utf-8 -*-
"""为**不在** eval_ncm_matrix.MAIN_TAGS 里的 run 算 NCM 矩阵（复用同一个 evaluate_one）。

用途：主表 run 之外的诊断/探针 run（如版本漂移诊断的 `vdc_seq`/`vdc_olora`）
也需要与论文完全同口径的 NCM 数字时，用本脚本，而不要去改 MAIN_TAGS
——那会顺带改变画图脚本的输入集合。

写到哪里由下方 OUT 决定；刻意与论文的 reports/ncm_matrix/ 分开，
避免把非主表 run 混进画图的产物目录。

⚠️ 与 eval_ncm_matrix.py 相同的语义提醒：`acc_cil[t][j]` 是由**终态模型**的
一份特征按累积类均值算出来的（checkpoint.pt 只存终态，无逐任务快照），
不是「任务 t 时的模型」在其测试集上的准确率。详见
`docs/实验产物溯源.md` 第 7 节。

用法：`python -m scripts.eval_ncm_matrix_extra`（无参数）。
"""
import json
import sys
from pathlib import Path

import torch

from peft_cl.utils.io import atomic_write_json
from scripts.eval_ncm import evaluate_one
from scripts.eval_ncm_sweep import discover_runs

BENCH = "imagenetr"
OUT = Path("reports/ncm_matrix_vdc") / BENCH

TARGETS = {("seq", "vdc_seq"), ("olora", "vdc_olora")}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = discover_runs(Path("experiments"), BENCH, ["seq", "olora"])

    picked = [(m, t, s, d) for (m, t, s, d, _x) in runs if (m, t) in TARGETS]
    if not picked:
        print("没找到 vdc run —— 检查 experiments/imagenetr/{seq,olora}/vdc_*/seed*/")
        return 1

    for method, tag, seed, seed_dir in sorted(picked):
        out = OUT / f"{method}__{tag}__seed{seed}.json"
        print(f"[run] {method}/{tag}/seed{seed}", flush=True)
        res = evaluate_one(BENCH, 20, seed, device, run_dir=str(seed_dir),
                           batch_size=64, num_workers=0)
        rec = {
            "benchmark": BENCH, "method": method, "tag": tag, "seed": seed,
            "num_tasks": len(res["acc_cil"]),
            "acc_cil": res["acc_cil"],
            "final_acc_cil": res["final_acc_cil"],
            "forgetting_cil": res["forgetting_cil"],
            "incremental_acc_cil": res["incremental_acc_cil"],
        }
        # 与 eval_ncm_sweep 一致：把 adapter 覆盖自检一并落盘（旧缓存无此字段）。
        if "load_audit" in res:
            rec["load_audit"] = res["load_audit"]
        atomic_write_json(out, rec)
        print(f"      NCM T=1 = {res['acc_cil'][0][0]*100:.2f}   "
              f"NCM final = {res['final_acc_cil']*100:.2f}", flush=True)

    print("\n已写入", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
