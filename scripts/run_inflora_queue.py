"""InfLoRA 基线队列（断点续训友好，幂等）。

论文的 Related Work 引用了 InfLoRA 并在实验里与之比较，因此必须真实跑出来。
本脚本负责把 InfLoRA 在「与其它方法完全相同的架构与超参」下跑满：

- Split CIFAR-100，20 任务，5 epochs/任务，seed 0..4（5 seeds）
- Split ImageNet-R，20 任务，5 epochs/任务，seed 0..2（3 seeds）

**断点续训**：每个 run 用 `run_single --resume` 调用；trainer 每学完一个任务就把
「可训参数 + 方法状态 + RNG 状态 + 类顺序 + 准确率矩阵」原子落盘。断电/被杀后
重跑本脚本即可从「上一个已完成任务 + 1」继续，不损失已完成的任务。

**幂等**：results.json 标记 finished 的 run 直接跳过。

用法（项目根目录，建议 detached + 日志）：
  python -u -m scripts.run_inflora_queue > reports/inflora_queue.log 2>&1
"""

import subprocess
import sys
import time
from pathlib import Path

# (benchmark, num_tasks, seed)
RUNS = (
    [("cifar100", 20, s) for s in range(5)]
    + [("imagenetr", 20, s) for s in range(3)]
)

METHOD = "inflora"
# tag 必须与 run_prompt_baselines.py 一致用空串（config.tag=""），由
# peft_cl.utils.paths.run_dir 统一映射到目录名 "default"。
# 若这里写 "default"，落盘目录仍是 default/，但 config.tag 变成 "default"——
# 同一类 run 在不同脚本里 tag 记法不同，会给按 tag 分桶的对账埋下陷阱。
TAG = ""
EPOCHS = 5

MAX_ATTEMPTS = 6      # 单个 run 连续失败多少次后放弃（避免死循环）
RETRY_SLEEP = 60      # 失败后等待秒数（给 OOM/休眠留恢复时间）


def run_dir_of(benchmark: str, seed: int) -> Path:
    # 与 run_prompt_baselines.run_dir_of 同一口径：空 tag 落到目录名 default。
    return Path("experiments") / benchmark / METHOD / (TAG or "default") / f"seed{seed}"


def is_finished(benchmark: str, seed: int) -> bool:
    rj = run_dir_of(benchmark, seed) / "results.json"
    if not rj.exists():
        return False
    import json
    try:
        return bool(json.loads(rj.read_text(encoding="utf-8")).get("finished"))
    except Exception:
        return False


def main():
    total = len(RUNS)
    for idx, (benchmark, num_tasks, seed) in enumerate(RUNS, 1):
        if is_finished(benchmark, seed):
            print(f"[{idx}/{total}] 跳过（已完成）{benchmark} seed{seed}", flush=True)
            continue
        cmd = [
            sys.executable, "-u", "-m", "scripts.run_single",
            "--benchmark", benchmark, "--num_tasks", str(num_tasks),
            "--method", METHOD, "--seed", str(seed), "--epochs", str(EPOCHS),
            "--resume",
        ]
        # 空 tag 时不传 --tag（与 run_prompt_baselines 同写法），让 run_single 的
        # 默认 tag="" 生效，从而 config.tag 与目录名 default 的映射统一。
        if TAG:
            cmd += ["--tag", TAG]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if is_finished(benchmark, seed):
                break
            print(f"[{idx}/{total}] 启动 {benchmark} seed{seed} "
                  f"(第 {attempt}/{MAX_ATTEMPTS} 次尝试)", flush=True)
            started = time.time()
            proc = subprocess.run(cmd)
            dt = time.time() - started
            if is_finished(benchmark, seed):
                print(f"[{idx}/{total}] 完成 {benchmark} seed{seed}，用时 {dt/3600:.2f} h",
                      flush=True)
                break
            print(f"[{idx}/{total}] 未完成（退出码 {proc.returncode}，{dt/60:.1f} min），"
                  f"{RETRY_SLEEP}s 后续训", flush=True)
            time.sleep(RETRY_SLEEP)
        else:
            print(f"[{idx}/{total}] !! 放弃 {benchmark} seed{seed}（重试上限）", flush=True)

    remaining = [(b, s) for b, _, s in RUNS if not is_finished(b, s)]
    print(f"\nInfLoRA 队列结束。未完成: {remaining if remaining else '无，全部完成'}", flush=True)


if __name__ == "__main__":
    main()
