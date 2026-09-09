"""P0 试点守护：循环 run_pilot20 直到 6 个 pilot20 实验全部 finished（断电自动续训）。

与 scripts/supervise_queue.py 同构：外层反复调 run_pilot20（内部每个 run_single 带
--resume、幂等跳过已完成），解决两类静默死：
1. 单个 run_single 子进程被 OOM/休眠杀 → run_pilot20 返回后 remaining() 仍 >0，本守护
   本轮 sleep 后重跑，幂等续上；
2. 守护父进程被杀（会话结束/笔记本关机）→ 由 detached 启动器或监控 cron 重启守护，
   重跑后从 checkpoint 续训。

用法（项目根目录，detached 启动）：python -u -m scripts.supervise_pilot20
"""
import glob
import json
import subprocess
import sys
import time

METHODS = {"seq", "ewc", "olora", "l2p", "coda", "folora_v2"}


def remaining() -> set:
    done = set()
    for p in glob.glob("experiments/cifar100/*/pilot20/seed0/results.json"):
        try:
            if json.load(open(p, encoding="utf-8")).get("finished"):
                done.add(p.split("/")[2])
        except (json.JSONDecodeError, OSError):
            pass
    return METHODS - done


def main() -> None:
    while True:
        rem = remaining()
        if not rem:
            print("SUPERVISOR: 6 个 pilot20 全部 finished，退出。", flush=True)
            return
        print(f"SUPERVISOR: 剩余 {len(rem)} 个 {sorted(rem)}，运行 run_pilot20。", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "scripts.run_pilot20"])
        time.sleep(30)


if __name__ == "__main__":
    main()
