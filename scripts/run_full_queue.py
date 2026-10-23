"""全量剩余队列编排：seed 对齐 → EWC 调优 → prompt 基线 → aggregate。

顺序（每阶段幂等，内部自动跳过已完成的实验、对未完成的自动 --resume 续训）：
1. seed 对齐（scripts.run_seed_align）：补齐 seq/ewc/olora/folora_v2 到 5 seed
   （cifar100 只剩 olora seed3 续训 + seq/ewc/olora seed4；imagenetr 全 4 方法 seed3/4）。
2. EWC 调优（scripts.run_ewc_tuned）：λ=30 补 seed1-4。
3. prompt 基线（scripts.run_prompt_baselines）：L2P/CODA × 2 基准 × 3 seed。
4. aggregate（scripts.aggregate）：汇总 cifar100 + imagenetr 主表数字。

用法（项目根目录）：python -u -m scripts.run_full_queue > reports/full_queue.log 2>&1
"""
import subprocess
import sys


STEPS = [
    ("seed_align", "scripts.run_seed_align"),
    ("ewc_tuned", "scripts.run_ewc_tuned"),
    ("prompt_baselines", "scripts.run_prompt_baselines"),
]


def main():
    for name, module in STEPS:
        print(f"\n========== 阶段开始：{name} ==========", flush=True)
        subprocess.run([sys.executable, "-u", "-m", module])
        print(f"========== 阶段结束：{name} ==========", flush=True)

    for bench in ["cifar100", "imagenetr"]:
        print(f"\n========== aggregate {bench} ==========", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "scripts.aggregate",
                        "--benchmark", bench])
    print("\n========== 全量队列完成 ==========", flush=True)


if __name__ == "__main__":
    main()
