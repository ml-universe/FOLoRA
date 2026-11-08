# -*- coding: utf-8 -*-
"""等训练链**真正**结束：连续 N 分钟没有任何 `run_lambda_grid` / `run_single` 进程。

为什么不直接 `wait-pid`：这些补跑批是**另一条** shell 里起的，它们的父 pid 在本进程
里拿不到；而按「日志里出现某行」判完成，一旦那条 shell 被关掉就永远等不到。
按进程判是唯一与启动方式无关的证据。

也不会误判中途换批：`run_lambda_grid` 是整批的父进程，批与批之间由
`chain_next_batch` 持有，父进程消失 15 分钟以上才认为整条链走完。

退出码 0 = 已空闲；1 = 超时（仍有人在跑），调用方应放弃而不是硬上。
"""
import subprocess
import sys
import time

IDLE_MIN = 15
MAX_HOURS = 240

PS = (
    "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "Where-Object { $_.CommandLine -match 'run_lambda_grid|run_single' }).Count"
)


def busy_count() -> int:
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                         capture_output=True, text=True, timeout=120)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        # 查不出来时**当成忙**：宁可多等，也不要在链还在跑时插队把显存挤爆
        return 1


def main() -> int:
    t0 = time.time()
    idle_since = None
    while True:
        n = busy_count()
        if n == 0:
            idle_since = idle_since or time.time()
            if (time.time() - idle_since) / 60 >= IDLE_MIN:
                print(f"[{time.strftime('%F %T')}] GPU 训练进程已空闲 "
                      f"{IDLE_MIN} 分钟，可以开跑")
                return 0
        else:
            if idle_since is not None:
                print(f"[{time.strftime('%F %T')}] 又有 {n} 个训练进程，重新计时")
            idle_since = None
        if (time.time() - t0) / 3600 > MAX_HOURS:
            print(f"[{time.strftime('%F %T')}] 等待超过 {MAX_HOURS} 小时，放弃")
            return 1
        print(f"[{time.strftime('%F %T')}] 训练进程 {n} 个，继续等", flush=True)
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
