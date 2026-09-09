"""最终强化实验（用户拍板：1a + 2 全做）。

两组目标（幂等 + --resume，可断点续训）：
- Group A（1a）：l2p/coda 用 20 epoch（prompt 基线各自最优），3 seed × 2 基准。
  cifar100 seed0 已在 P0 试点完成（tag=pilot20），跳过。
- Group B（2）：CIFAR-100 主基线 seq/ewc/olora/folora_v2 补 seed5-9（5→10 seed），
  做实显著性（10 seed = 10 个 class order，直接压方差）。
"""
import subprocess
import sys

# (bench, method, tag, seed, epochs, extra_kwargs)
TARGETS = []

# Group B：seq/ewc/olora/folora_v2，cifar100，seed 5-9（10-seed 显著性）
for method, tag, extra in [
    ("seq", "default", None),
    ("ewc", "default", None),
    ("olora", "default", None),
    ("folora_v2", "v2f_l300_k16", {"folora_lambda": 300, "folora_topk": 16}),
]:
    for seed in range(5, 10):
        TARGETS.append(("cifar100", method, tag, seed, 5, extra))

# Group A：l2p/coda 20 epoch（公平 prompt 基线），3 seed × 2 基准
for bench in ["cifar100", "imagenetr"]:
    for method in ["l2p", "coda"]:
        for seed in range(3):
            if bench == "cifar100" and seed == 0:
                continue  # P0 试点已完成 cifar100 seed0（tag=pilot20）
            TARGETS.append((bench, method, "pilot20", seed, 20, None))


def run_one(bench, method, tag, seed, epochs, extra) -> None:
    cmd = [sys.executable, "-u", "-m", "scripts.run_single",
           "--method", method, "--seed", str(seed), "--resume",
           "--benchmark", bench, "--num_tasks", "20",
           "--epochs", str(epochs), "--lora_rank", "16",
           "--tag", tag]
    if extra:
        for k, v in extra.items():
            cmd += [f"--{k}", str(v)]
    print(f"[boost] {bench} {method} seed{seed} epochs{epochs} tag={tag}", flush=True)
    subprocess.run(cmd)


def main() -> None:
    for t in TARGETS:
        run_one(*t)


if __name__ == "__main__":
    main()
