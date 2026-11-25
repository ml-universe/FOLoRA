"""在一个基准上做**基于验证集**的超参重选（回应 R8-6）。

审稿人的质疑
------------
R8 指出：主配置是靠**测试集**挑出来的（每个基准都取网格里测试分数最高的点），
没有留出验证集；因此「主配置 k=64, λ=3」这个说法仍可能被读成事后挑参。R8-6 要求
「至少在一个基准上用验证集重选一次」。

本脚本做的事
------------
1. 逐类留出 `--val_frac` 比例的**训练**样本（`CLConfig.val_frac`，见
   `src/peft_cl/data/split.py::ContinualSplit._split_val`）。留出索引由
   (seed, task, class) 确定性生成 ⇒ 同一 seed 下所有候选配置看到**同一批**留出样本，
   换 λ 只换训练数据的使用方式，不换验证数据。
2. 对 `--lambdas` 里每个 λ × `--seeds` 里每个 seed 训练一次（其余超参固定），
   训练**从未见过**留出样本。
3. 用**与主表完全相同的 NCM 口径**（复用 `utils.ncm_trace.ncm_row`）分别算
   ① 留出验证集上的平均准确率（选参用）② 官方测试集上的平均准确率（报告用）。
   验证集与测试集都走 eval transform，且原型都由「去掉留出样本后的训练集」构造，
   两侧口径一致。
4. 按验证集均值选 λ，报告该 λ 的测试均值和全部候选的测试均值，写
   `reports/val_selection.json`。

用法（项目根目录）：
    py -3.11 -m scripts.val_select --benchmark cifar100 \
        --lambdas 1 3 10 --seeds 0 1 2 --val_frac 0.1
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from peft_cl.data.datasets import get_transform
from peft_cl.trainer import CLTrainer
from peft_cl.utils.config import CLConfig
from peft_cl.utils.ncm_trace import ncm_row
from peft_cl.utils.paths import run_dir

BENCHMARK_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenetr": 200}


class _ValSplitView:
    """把 ContinualSplit 的「test」换成「留出的验证集」，其余原样转发。

    这样可以直接复用 `ncm_row()`：它按 `split.task_dataset(j, "test")` 取评估数据、
    按 `split.task_dataset(j, "train")` 取原型。于是验证集评估与主表口径逐字相同，
    只是测试集换成了从未参与训练的那部分样本。
    """

    def __init__(self, split):
        self._split = split

    def task_dataset(self, task_id: int, which: str):
        if which == "test":
            return self._split.val_dataset(task_id)
        return self._split.task_dataset(task_id, which)


def _ncm_mean(model, split_like, num_tasks: int, classes_per_task: int, device,
              num_workers: int = 0) -> float:
    """终态模型在 split_like 的「test」上的 NCM 平均准确率（与主表同口径）。"""
    row_cil, _ = ncm_row(model, split_like, num_tasks - 1, classes_per_task,
                         num_tasks, device, batch_size=64,
                         num_workers=num_workers)
    return float(np.mean(row_cil))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="基于留出验证集的超参重选")
    p.add_argument("--benchmark", default="cifar100",
                   choices=["cifar10", "cifar100", "imagenetr"])
    p.add_argument("--num_tasks", type=int, default=20)
    p.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 3.0, 10.0],
                   help="候选正则强度（默认 1/3/10，即主文的 λ 网格）")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="逐类留出的训练样本比例（默认 0.1）")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--folora_topk", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="experiments")
    p.add_argument("--out", default="reports/val_selection.json")
    return p


def _cfg(args, lam: float, seed: int) -> CLConfig:
    tag = f"valsel_l{lam:g}_k{args.folora_topk}"
    return CLConfig(
        benchmark=args.benchmark, num_tasks=args.num_tasks,
        method="folora_v2", seed=seed, epochs=args.epochs,
        lora_rank=args.lora_rank, lora_alpha=args.lora_rank,
        folora_lambda=lam, folora_topk=args.folora_topk,
        val_frac=args.val_frac,
        tag=tag, data_root=args.data_root, out_dir=args.out_dir,
    )


def main() -> None:
    args = build_parser().parse_args()
    num_classes = BENCHMARK_CLASSES[args.benchmark]
    classes_per_task = num_classes // args.num_tasks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():                      # 断点续跑：已算过的 (λ, seed) 不重跑
        with open(out_path, encoding="utf-8") as f:
            records = json.load(f).get("runs", [])
    done = {(float(r["lambda"]), int(r["seed"])) for r in records}

    for lam in args.lambdas:
        for seed in args.seeds:
            if (float(lam), int(seed)) in done:
                print(f"[skip] λ={lam:g} seed={seed} already in {out_path}")
                continue
            cfg = _cfg(args, lam, seed)
            rdir = run_dir(cfg)
            print(f"[run] λ={lam:g} seed={seed} -> {rdir}")
            trainer = CLTrainer(cfg, resume=True)          # 有 checkpoint 就续跑/直接收尾
            trainer.run()

            split = trainer.split
            # 验证/测试都走 eval transform：留出样本不参与训练，两侧口径一致，
            # 原型也由同一（无随机翻转的）视角构造。
            saved_tf = split.base_train.transform
            split.base_train.transform = get_transform(cfg.image_size, False)
            try:
                val_acc = _ncm_mean(trainer.model, _ValSplitView(split),
                                    cfg.num_tasks, classes_per_task, device)
                test_acc = _ncm_mean(trainer.model, split,
                                     cfg.num_tasks, classes_per_task, device)
            finally:
                split.base_train.transform = saved_tf

            rec = {"lambda": float(lam), "seed": int(seed),
                   "val_acc": val_acc, "test_acc": test_acc,
                   "val_frac": args.val_frac, "run_dir": str(rdir)}
            records.append(rec)
            print(f"      val={val_acc:.4f}  test={test_acc:.4f}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"runs": records}, f, indent=2, ensure_ascii=False)

            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- 汇总：按验证集均值选 λ，报该 λ 的测试均值 ----
    # 汇总覆盖 JSON 里所有已算过的 λ（而不只是本次命令行传入的），
    # 这样每次只跑一个 (λ, seed) 的调用也不会把 by_lambda / selected_lambda 写残。
    summary = {}
    for lam in sorted({float(r["lambda"]) for r in records}):
        vs = [r["val_acc"] for r in records if float(r["lambda"]) == float(lam)]
        ts = [r["test_acc"] for r in records if float(r["lambda"]) == float(lam)]
        if vs:
            summary[f"{lam:g}"] = {
                "n": len(vs),
                "val_mean": float(np.mean(vs)), "val_std": float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
                "test_mean": float(np.mean(ts)), "test_std": float(np.std(ts, ddof=1)) if len(ts) > 1 else 0.0,
            }
    selected = max(summary, key=lambda k: summary[k]["val_mean"]) if summary else None
    per_seed_pick = {}
    for seed in sorted({int(r["seed"]) for r in records}):
        cand = [r for r in records if int(r["seed"]) == seed]
        if cand:
            per_seed_pick[str(seed)] = f"{max(cand, key=lambda r: r['val_acc'])['lambda']:g}"

    result = {"benchmark": args.benchmark, "val_frac": args.val_frac,
              "runs": records, "by_lambda": summary,
              "selected_lambda": selected,
              "per_seed_pick": per_seed_pick}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ("by_lambda", "selected_lambda",
                                             "per_seed_pick")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
