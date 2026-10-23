"""单次实验入口（断点续训友好）。

用法（在项目根目录）：
  # 论文主配置（FOLoRA v2）：
  python -m scripts.run_single --benchmark cifar100 --method folora_v2 --seed 0 \
      --lora_rank 64 --folora_lambda 3 --folora_topk 64 --tag v2f_l3_k64
  # 续跑一个已有的 run：
  python -m scripts.run_single --benchmark cifar100 --method folora_v2 --seed 0 \
      --lora_rank 64 --folora_lambda 3 --folora_topk 64 --tag v2f_l3_k64 --resume

注意 `--method` 的默认值是已废弃的 v1（`folora`），本脚本会**拒绝**新起 v1 run
（须显式 `--allow-legacy-v1`）；续跑不受限。

说明：
- `python -m scripts.xx` 需在项目根目录执行（cwd 会进 sys.path），并已 `pip install -e .`。
- 脚本幂等：若 results.json 已标记 finished，直接跳过退出（网格调度器依赖这一点）。
- 断电恢复：加 --resume 才会从 checkpoint 续跑（不加以 resume 会从头覆盖）。
"""

import argparse
import json

from peft_cl.trainer import CLTrainer
from peft_cl.utils.config import CLConfig
from peft_cl.utils.paths import run_dir


def build_parser() -> argparse.ArgumentParser:
    """本脚本 CLI 的**唯一事实来源**。

    抽成函数是为了让 `scripts/retrain_fix1.py::build_cmd` 能在生成命令前查每个选项到底是
    `store_true`（不带值）还是要带值 —— 2026-10-04 就因为它在那边靠 `isinstance(v, bool)`
    猜，把 `--folora_weighted 1` 生成成了裸 `--folora_weighted`，argparse 直接 exit 2，
    22/35 个重训 run 一个都没跑起来（详见 retrain_fix1.py 的 build_cmd 注释）。
    """
    p = argparse.ArgumentParser(description="运行一次持续学习实验")
    p.add_argument("--benchmark", default="cifar100",
                   choices=["cifar10", "cifar100", "imagenetr"])
    p.add_argument("--num_tasks", type=int, default=20)
    p.add_argument("--method", default="folora",
                   choices=["seq", "ewc", "olora", "inflora", "folora", "folora_v2", "l2p", "coda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--folora_lambda", type=float, default=300.0)
    p.add_argument("--folora_topk", type=int, default=0)
    p.add_argument("--folora_weighted", type=int, choices=[0, 1], default=1,
                   help="1=按 Fisher 特征值加权（默认）；0=等权消融（权重齐次化，同一 λ 可直接比）")
    p.add_argument("--olora_aggregate", choices=["sum", "mean"], default="sum",
                   help="O-LoRA 推理聚合：sum=论文原式 Σ B_i A_i（默认）；"
                        "mean=除以任务数，公平性对照（见 multi_lora.py）")
    p.add_argument("--olora_orth_lambda", type=float, default=0.0,
                   help="O-LoRA 正交性约束 λ₁（原文目标里的正则项）。"
                        "0=只正交初始化、不施加训练期约束（=已完成那批 run 的语义）；"
                        "建议 0.1 起手，先验掉得多就往 0.5~1.0 加")
    p.add_argument("--ewc_lambda", type=float, default=100.0)
    p.add_argument("--prompt_pool_size", type=int, default=20)
    p.add_argument("--prompt_length", type=int, default=5)
    p.add_argument("--prompt_topk", type=int, default=5)
    p.add_argument("--fisher_batches", type=int, default=300)
    p.add_argument("--tag", default="")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="experiments")
    p.add_argument("--resume", action="store_true", help="从 checkpoint 续训")
    p.add_argument("--allow-legacy-v1", action="store_true",
                   help="显式允许用**已被取代的 v1 实现**（--method folora）新起一个 run。"
                        "默认拒绝——见 main() 里的说明。已有的 v1 run 仍可用 --resume 续跑。")
    p.add_argument("--bound_probe", action="store_true",
                   help="在每个任务边界测「界的二阶项 vs 实测遗忘」（验证 04_theory.tex "
                        "的高阶项承诺），结果写 run 目录的 bound_terms.jsonl。默认关。")
    return p


def main():
    p = build_parser()
    args = p.parse_args()

    # 拦住「新起一个 v1 run」。v1（method="folora"）的正则项是退化实现、对训练近似
    # no-op，跑出来的数字看着合理但不是论文方法（论文是 folora_v2）。忘写 --method 时
    # 默认值恰好是它，所以这里宁可报错也不让它静默跑完。
    # 注意**不拦 --resume**：已有的 v1 run 还要能续跑（那是证据，不能因为拦新 run 而作废），
    # 也不影响 eval_ncm 读取旧 run 的 config.json（那条路径不经过本脚本）。
    if args.method == "folora" and not args.resume and not args.allow_legacy_v1:
        p.error(
            "--method folora 是已被取代的 v1 实现（正则项退化、对训练近似 no-op），"
            "论文用的是 --method folora_v2（主配置 --folora_lambda 3 --folora_topk 64 "
            "--lora_rank 64）。若要续跑已有的 v1 run，加 --resume；"
            "若确实要新起一个 v1 run，加 --allow-legacy-v1。")

    cfg = CLConfig(
        benchmark=args.benchmark, num_tasks=args.num_tasks, method=args.method,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        folora_lambda=args.folora_lambda, folora_topk=args.folora_topk,
        folora_weighted=bool(args.folora_weighted),
        olora_aggregate=args.olora_aggregate,
        olora_orth_lambda=args.olora_orth_lambda,
        ewc_lambda=args.ewc_lambda,
        prompt_pool_size=args.prompt_pool_size, prompt_length=args.prompt_length,
        prompt_topk=args.prompt_topk,
        fisher_batches=args.fisher_batches,
        tag=args.tag, data_root=args.data_root, out_dir=args.out_dir,
        bound_probe=args.bound_probe,
    )

    # 幂等：已完成则直接退出
    results_path = run_dir(cfg) / "results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            if json.load(f).get("finished"):
                print(f"[skip] already finished: {results_path}")
                return

    CLTrainer(cfg, resume=args.resume).run()


if __name__ == "__main__":
    main()
