# -*- coding: utf-8 -*-
"""A10：峰值显存的**实测**（R1 资源表要的那一列）。

背景
----
回复信此前一直用**解析记账**（`scripts/memory_footprint.py`：重要性状态 195.2 MB 等）回答
显存问题。解析值回答的是「方法自己额外存了什么」，**不是 GPU 实际峰值**——审稿人若追问
"peak device memory"，解析值答不上。本脚本补的就是实测：每个方法在训练时真正吃到的
`torch.cuda.max_memory_allocated()`。

为什么每种方法**各起一个子进程**
--------------------------------
CUDA 的 caching allocator 在进程内**不归还**显存：同一进程里先跑 O-LoRA（保留 20 个
per-task adapter）再跑 Seq，Seq 的峰值会读到 O-LoRA 留下的缓存，数字全是错的。子进程
退出即释放，才有可比性。

为什么记**逐任务**的峰值而不只记一个总数
----------------------------------------
O-LoRA / InfLoRA 的显存是**随任务数增长**的（每任务一个模块），只报终态峰值会把「增长」
这个性质藏起来；逐任务曲线才能说明它们与前缀方法的差别。

口径
----
* 20 任务、**每任务 1 epoch**（峰值由「模型 + 一个 batch + 优化器状态」决定，与 epoch 数
  无关；1 epoch 把每个方法压到约 15 分钟）。主表用 5 epoch，这一点必须写进回复信。
* 与主表同超参：FOLoRA 用 λ=3 / k=64 / rank 64，EWC 用 λ=1000，O-LoRA / InfLoRA 用 rank 16。
* 结果写 `reports/peak_memory.json` 与 `reports/peak_memory/<method>.json`；
  训练产物落在 `experiments_peakmem/`（**故意不与 `experiments/` 混**，避免被聚合脚本
  当成主表 run）。

用法
----
  py -3.11 -m scripts.measure_peak_mem                 # 跑全部五种方法并汇总
  py -3.11 -m scripts.measure_peak_mem --method ewc    # 只跑一种（内部就是这么调的）
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

METHODS = ["seq", "ewc", "olora", "inflora", "folora_v2"]

# 与主表一致的超参；没列出的方法用 CLConfig 默认值
EXTRA = {
    "seq": [],
    "ewc": ["--ewc_lambda", "1000"],
    "olora": ["--lora_rank", "16"],
    "inflora": ["--lora_rank", "16"],
    "folora_v2": ["--lora_rank", "64", "--folora_topk", "64", "--folora_lambda", "3"],
}

OUT_DIR = Path("reports/peak_memory")
OUT_ROOT = Path("experiments_peakmem")


def run_one(method: str, num_tasks: int, epochs: int,
            overrides: dict = None, tag: str = "a10") -> dict:
    """在**本进程**里跑一种方法并实测峰值。

    调用方保证它是被独立子进程拉起的（见模块 docstring「为什么各起一个子进程」）。
    `overrides` 用于归因实验：只改一个旋钮（例如把 FOLoRA 的 rank 降到 16 与基线对齐），
    看峰值跟着谁变，从而把「秩」与「方法本身」分开。
    """
    import torch

    from peft_cl.trainer import CLTrainer
    from peft_cl.utils.config import CLConfig

    extra = list(EXTRA[method])
    if overrides:
        for k, v in overrides.items():
            extra += [f"--{k}", str(v)]
    kw = {}
    for i in range(0, len(extra), 2):
        kw[extra[i].lstrip("-")] = extra[i + 1]
    for k in ("lora_rank", "folora_topk"):
        if k in kw:
            kw[k] = int(kw[k])
    if "folora_lambda" in kw:
        kw["folora_lambda"] = float(kw["folora_lambda"])
    if "ewc_lambda" in kw:
        kw["ewc_lambda"] = float(kw["ewc_lambda"])

    cfg = CLConfig(benchmark="cifar100", num_tasks=num_tasks, epochs=epochs,
                   method=method, seed=0, tag=tag, out_dir=str(OUT_ROOT), **kw)
    trainer = CLTrainer(cfg)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # 重要性估计（Fisher / 对角 Fisher / no-op）的耗时：包住 before_task 而不是
    # 单独计时某个函数——不同方法在这里做的事不同，只有 before_task 是共同入口。
    imp = {"seconds": 0.0, "calls": 0}
    orig_before = trainer.method.before_task

    def timed_before(*a, **k):
        t0 = time.perf_counter()
        out = orig_before(*a, **k)
        torch.cuda.synchronize()          # 计时前必须同步，否则只量到 kernel 入队
        imp["seconds"] += time.perf_counter() - t0
        imp["calls"] += 1
        return out

    trainer.method.before_task = timed_before

    per_task = []
    orig_train = trainer._train_task

    def train_timed(task_id: int):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        orig_train(task_id)
        per_task.append({
            "task": task_id,
            "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
            "wall_s": round(time.perf_counter() - t0, 1),
        })

    trainer._train_task = train_timed
    t0 = time.perf_counter()
    trainer.run()
    wall = time.perf_counter() - t0

    res = {
        "method": method,
        "num_tasks": num_tasks,
        "epochs_per_task": epochs,
        "config": cfg.to_dict(),
        "peak_alloc_mib": max(t["peak_alloc_mib"] for t in per_task),
        "peak_reserved_mib": max(t["peak_reserved_mib"] for t in per_task),
        "final_alloc_mib": round(torch.cuda.memory_allocated() / 2 ** 20, 1),
        "importance_seconds_total": round(imp["seconds"], 1),
        "importance_calls": imp["calls"],
        "importance_seconds_per_task": (round(imp["seconds"] / imp["calls"], 1)
                                        if imp["calls"] else None),
        "wall_seconds": round(wall, 1),
        "per_task": per_task,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="A10 峰值显存实测")
    ap.add_argument("--method", default=None, choices=METHODS,
                    help="只跑这一种（内部使用）")
    ap.add_argument("--num_tasks", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1)
    # 冒烟：只建 trainer、打补丁、不训练。用于在 GPU 被占用时验证接口没接错
    # （CLConfig 字段名、method.before_task 可替换、_train_task 可替换、run_dir 路径）。
    # 真正的训练循环与 run_single 是同一段代码，接口对了就不会错。
    ap.add_argument("--smoke", action="store_true")
    # 归因实验用：--set lora_rank=16 --set folora_topk=16 --label k16
    # 只改指定的旋钮，其余仍走 EXTRA 里与主表一致的值。
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖 EXTRA 里的配置（可重复）；只能配合 --method 使用")
    ap.add_argument("--label", default=None,
                    help="结果写到 reports/peak_memory/<method>__<label>.json，"
                         "run 目录用 tag=a10_<label>，**不覆盖**正式的 a10 结果")
    args = ap.parse_args()

    if args.smoke:
        from peft_cl.trainer import CLTrainer
        from peft_cl.utils.config import CLConfig
        cfg = CLConfig(benchmark="cifar100", num_tasks=1, epochs=1, method="seq",
                       tag="a10_smoke", out_dir=str(OUT_ROOT))
        tr = CLTrainer(cfg)
        assert hasattr(tr, "method") and hasattr(tr.method, "before_task"), "method hook 缺失"
        tr.method.before_task = lambda *a, **k: None
        tr._train_task = lambda task_id: None
        print(f"SMOKE OK: run_dir={tr.dir}  method={type(tr.method).__name__}  "
              f"peak_alloc={__import__('torch').cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.method:
        overrides = {}
        for kv in args.set:
            k, _, v = kv.partition("=")
            overrides[k.strip()] = v.strip()
        tag = f"a10_{args.label}" if args.label else "a10"
        res = run_one(args.method, args.num_tasks, args.epochs,
                      overrides=overrides or None, tag=tag)
        name = f"{args.method}__{args.label}.json" if args.label else f"{args.method}.json"
        out = OUT_DIR / name
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{args.method}{'/' + args.label if args.label else ''}] "
              f"peak_alloc={res['peak_alloc_mib']} MiB "
              f"peak_reserved={res['peak_reserved_mib']} MiB "
              f"importance={res['importance_seconds_total']} s "
              f"({res['importance_calls']} calls) -> {out}", flush=True)
        return

    # 父进程：逐个拉起子进程。**失败不中断**——少一个方法的实测比整批作废好，
    # 汇总里 status 会如实标出来。
    summary = {"num_tasks": args.num_tasks, "epochs_per_task": args.epochs,
               "methods": {}, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for m in METHODS:
        cmd = [sys.executable, "-u", "-m", "scripts.measure_peak_mem",
               "--method", m, "--num_tasks", str(args.num_tasks),
               "--epochs", str(args.epochs)]
        print(f"=== {m} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        p = OUT_DIR / f"{m}.json"
        if rc == 0 and p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            summary["methods"][m] = {k: d[k] for k in
                                     ("peak_alloc_mib", "peak_reserved_mib",
                                      "final_alloc_mib", "importance_seconds_total",
                                      "importance_calls", "importance_seconds_per_task",
                                      "wall_seconds")}
        else:
            summary["methods"][m] = {"status": f"failed rc={rc}"}
            print(f"[{m}] 失败，退出码 {rc}", flush=True)

    ok = [v["peak_alloc_mib"] for v in summary["methods"].values()
          if "peak_alloc_mib" in v]
    if ok:
        summary["peak_alloc_mib_range"] = [min(ok), max(ok)]
    Path("reports/peak_memory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 汇总 ===")
    for m, v in summary["methods"].items():
        if "peak_alloc_mib" in v:
            print(f"{m:10s} peak {v['peak_alloc_mib']:8.1f} MiB  "
                  f"importance {v['importance_seconds_total']:6.1f} s "
                  f"({v['importance_calls']} calls)")
        else:
            print(f"{m:10s} {v['status']}")
    print("写入 reports/peak_memory.json")


if __name__ == "__main__":
    main()
