# -*- coding: utf-8 -*-
"""R2 的 GPU 尾巴：等训练链彻底空闲后，按顺序补跑三件与 R1-2 有关的证据。

为什么要串成一条队列
--------------------
2026-11-24 起主表的 O-LoRA 从「只做正交初始化」(`default_fix1`) 换成了「训练期正交性
目标 λ₁=1.0」(`olora_orth_l1_fix1`)。主表的数字、显著性家族、TOST 基线、RanPAC 后验表
都已经切过去了，但**图 2/3 的候选类数矩阵**和**后验表的 RanPAC 列**还停在旧 tag 上。
两者都要重新提特征（要 GPU），而此刻 L2-SP 的 ImageNet-R 批还在跑，显存只有 8 GB，
只能排队等它结束。

顺序有讲究：先矩阵（便宜、决定图 2/3 能不能重画），再画图，再 RanPAC（最贵），
最后才是 `val_select`（9 个 CIFAR run，R8-6 的验证集重选，最久，放最后不挡别人）。

用法：
    py -3.11 scripts/launch_detached.py --log logs/r2_gpu_queue.log -- \
        -m scripts.r2_gpu_queue
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

IDLE_MIN = 10
REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

PS = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "Where-Object { $_.CommandLine -match 'run_lambda_grid|run_single|val_select' }).Count")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def busy_count() -> int:
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                         capture_output=True, text=True, timeout=120)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 1          # 查不出来就当忙：宁可多等，也别把显存挤爆


def wait_idle() -> None:
    idle_since = None
    while True:
        n = busy_count()
        if n == 0:
            idle_since = idle_since or time.time()
            if (time.time() - idle_since) / 60 >= IDLE_MIN:
                log(f"训练进程已空闲 {IDLE_MIN} 分钟，开始队列")
                return
        else:
            idle_since = None
        log(f"训练进程 {n} 个，继续等")
        time.sleep(120)


def run(args: list[str], allow_fail: bool = False) -> bool:
    log("RUN " + " ".join(args))
    r = subprocess.run([PY] + args, cwd=REPO)
    if r.returncode != 0:
        log(f"失败（exit={r.returncode}）：{' '.join(args)}")
        if not allow_fail:
            return False
    return True


def main() -> int:
    wait_idle()

    steps: list[list[str]] = [
        # 1) 图 2/3 的候选类数矩阵（新 O-LoRA tag）
        ["-m", "scripts.eval_ncm_matrix", "--benchmark", "cifar100"],
        ["-m", "scripts.eval_ncm_matrix", "--benchmark", "imagenetr"],
        # 2) 用新矩阵重画图 2/3（缺任何一格都会报错，不会静默用旧图）
        ["-m", "scripts.plot_forgetting_curve", "--benchmark", "all"],
        # 3) 后验表的 RanPAC 列：按 seed 逐轮重跑 sweep（特征缓存会跳过没变的 run）
    ]
    #     `--out` 是必须的：不给的话结果只打到子进程的 stdout，队列日志里收不到
    #     （2026-11-24 第一版漏了，8 轮 sweep 白跑约 3 小时，只剩特征缓存还算数）。
    #     文件名用 `_main_` 而不是复用旧的 `_sweep_`：旧文件里是 `default_fix1`
    #     那批 run，覆盖掉就没有旧证据了；load_ranpac 按 (基准, 方法, tag) 归档，
    #     两份文件不会撞。
    for seed in range(5):
        steps.append(["-m", "scripts.eval_ranpac", "--benchmark", "cifar100",
                      "--sweep", "--seed", str(seed),
                      "--out", f"reports/ranpac_cifar100_main_seed{seed}.json"])
    for seed in range(3):
        steps.append(["-m", "scripts.eval_ranpac", "--benchmark", "imagenetr",
                      "--sweep", "--seed", str(seed),
                      "--out", f"reports/ranpac_imagenetr_main_seed{seed}.json"])
    # 4) 后验表重生成（--check 只是打印，这里直接写文件）
    steps.append(["-m", "scripts.make_posthoc_table"])
    # 5) R8-6：验证集重选（9 个 CIFAR run，最久，最后跑）
    steps.append(["-m", "scripts.val_select", "--benchmark", "cifar100",
                  "--lambdas", "1", "3", "10", "--seeds", "0", "1", "2"])

    for i, args in enumerate(steps, 1):
        log(f"---- 步骤 {i}/{len(steps)} ----")
        if not run(args):
            log("队列中止：前一步失败，后面的步骤依赖它，不做无意义的重试")
            return 1
    log("队列全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
