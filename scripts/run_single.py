"""单次实验入口（断点续训友好）。

用法（在项目根目录）：
  python -m scripts.run_single --benchmark cifar100 --method folora --seed 0
  python -m scripts.run_single --benchmark cifar100 --method folora --seed 0 --resume

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


def main():
    p = argparse.ArgumentParser(description="运行一次持续学习实验")
    p.add_argument("--benchmark", default="cifar100",
                   choices=["cifar10", "cifar100", "imagenetr"])
    p.add_argument("--num_tasks", type=int, default=20)
    p.add_argument("--method", default="folora",
                   choices=["seq", "ewc", "olora", "folora", "folora_v2", "l2p", "coda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--folora_lambda", type=float, default=300.0)
    p.add_argument("--folora_topk", type=int, default=0)
    p.add_argument("--ewc_lambda", type=float, default=100.0)
    p.add_argument("--prompt_pool_size", type=int, default=20)
    p.add_argument("--prompt_length", type=int, default=5)
    p.add_argument("--prompt_topk", type=int, default=5)
    p.add_argument("--fisher_batches", type=int, default=300)
    p.add_argument("--tag", default="")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="experiments")
    p.add_argument("--resume", action="store_true", help="从 checkpoint 续训")
    args = p.parse_args()

    cfg = CLConfig(
        benchmark=args.benchmark, num_tasks=args.num_tasks, method=args.method,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        folora_lambda=args.folora_lambda, folora_topk=args.folora_topk,
        ewc_lambda=args.ewc_lambda,
        prompt_pool_size=args.prompt_pool_size, prompt_length=args.prompt_length,
        prompt_topk=args.prompt_topk,
        fisher_batches=args.fisher_batches,
        tag=args.tag, data_root=args.data_root, out_dir=args.out_dir,
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
