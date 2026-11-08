"""校验 `utils/ncm_trace.ncm_row` 与主表 NCM 口径一致（跑 E2 前的闸门）。

原理
----
`eval_ncm_sweep` 用**终态** checkpoint 算出的 `final_acc_cil`，正是「在 θ_T 特征上、
对全部 20 个类均值、在已见全部类里做最近邻」的平均准确率。`ncm_row(task_id=19)` 在同
一个 checkpoint 上做的事**完全相同**（只是它被设计成可以在任意边界 t 调用）。所以两者的
数值必须一致；不一致就说明 trace 的特征提取、类均值或类号映射有偏差，**此时不能开跑**
——那会把 20 小时的 GPU 花在一套与主表不可比的曲线上。

用法
----
    py -3.11 scripts/validate_ncm_trace.py \\
        --method folora_v2 --tag v2f_l3_k64 --seed 0

退出码 0 = 通过；1 = 不一致（差值超过 --tol）。
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from peft_cl.trainer.trainer import CLTrainer
from peft_cl.utils.config import CLConfig
from peft_cl.utils.ncm_trace import ncm_row
from peft_cl.utils.paths import run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="cifar100")
    p.add_argument("--method", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-3,
                   help="允许的最大绝对差（比例，不是百分点）。默认 1e-3 = 0.1 点。")
    args = p.parse_args()

    cfg = CLConfig.load(str(Path("experiments") / args.benchmark / args.method /
                            (args.tag or "default") / f"seed{args.seed}" / "config.json"))
    trainer = CLTrainer(cfg, resume=True)
    last = cfg.num_tasks - 1
    row_cil, row_til = ncm_row(trainer.model, trainer.split, last,
                               trainer.classes_per_task, cfg.num_tasks, trainer.device)
    got = sum(row_cil) / cfg.num_tasks

    cache = (Path("reports") / "ncm" / args.benchmark /
             f"{args.method}__{args.tag or 'default'}__seed{args.seed}.json")
    if not cache.exists():
        print(f"[SKIP] 没有 NCM 缓存 {cache}，无法比对。先跑 eval_ncm_sweep。")
        sys.exit(2)
    ref = json.loads(cache.read_text(encoding="utf-8"))["final_acc_cil"]

    diff = got - ref
    ok = abs(diff) <= args.tol
    print(f"ncm_row(t={last}) = {got:.6f}")
    print(f"eval_ncm 缓存     = {ref:.6f}")
    print(f"差值              = {diff:+.6f}  (容差 {args.tol})")
    # 顺带打印 own-task 一列，便于肉眼看有没有整体性错位
    print(f"row_cil 前 3 项    = {[round(v, 4) for v in row_cil[:3]]}")
    print(f"row_til 前 3 项    = {[round(v, 4) for v in row_til[:3]]}")
    if not ok:
        print("!! 不一致。**不要**启动 ncm_trace 批次：先查特征 transform / 类号映射 / 均值归一化。")
        sys.exit(1)
    print("OK：trace 与主表同口径，可以开跑。")


if __name__ == "__main__":
    main()
