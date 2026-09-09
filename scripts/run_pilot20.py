"""P0 试点：验证 20 epoch 是否把精度从 ~10% 拉到有意义水平 + 基线是否变可信。

背景：当前全量实验用 5 epoch/任务，精度仅 ~10%、方差大、O-LoRA/L2P/CODA 在
ImageNet-R 塌缩到 seq 以下。文献标准是 20–30 epoch。本试点在 CIFAR-100 上把
6 个方法各跑 1 个 seed、20 epoch，快速验证「加 epoch」能否根治。

幂等 + --resume，可断点续跑。tag=pilot20 与 5-epoch 结果（default/v2f_l300_k16）
完全分离，不覆盖。
"""
import subprocess
import sys

METHODS = [
    ("seq", {}),
    ("ewc", {}),
    ("olora", {}),
    ("l2p", {}),
    ("coda", {}),
    ("folora_v2", {"folora_lambda": 300, "folora_topk": 16}),
]


def main() -> None:
    for method, extra in METHODS:
        cmd = [sys.executable, "-u", "-m", "scripts.run_single",
               "--method", method, "--seed", "0", "--resume",
               "--benchmark", "cifar100", "--num_tasks", "20",
               "--epochs", "20", "--lora_rank", "16",
               "--tag", "pilot20"]
        for k, v in extra.items():
            cmd += [f"--{k}", str(v)]
        print(f"[pilot20] {method} seed0", flush=True)
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
