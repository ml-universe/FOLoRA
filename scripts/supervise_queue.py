"""外层守护：循环运行 run_full_queue，直到「目标实验集」全部 finished=true。

背景：run_full_queue 编排 seed_align → ewc_tuned → prompt_baselines → aggregate，
各步幂等（跳过已完成、--resume 续训未完成）。但两类静默死亡会中断队列：
1. 单个 run_single 子进程被 OOM/休眠杀掉 → run_full_queue 的 subprocess.run 返回后
   继续下一个实验，被杀的那个未 finished；
2. run_full_queue 父进程本身被杀（会话结束/笔记本休眠）。

本守护是独立进程，反复调用 run_full_queue 直到目标集全 finished：
- 第 1 类死亡：run_full_queue 返回后，remaining() 仍 >0，本轮 sleep 后重跑，幂等续上；
- 第 2 类死亡：守护自身作为独立进程（外部 detached 启动）重启后继续，或由监控 cron 重启。

用法（项目根目录，detached 启动，日志追加）：
  python -u -m scripts.supervise_queue
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# 目标实验集 = 论文主表 + 调优 EWC + prompt 基线 需要的全部 finished 实验。
# （不含 EWC λ 扫描的 ewc_lam1/10/100/1000/300 及旧 folora/folora_v2 扫描残留——
#  那些仅用于调参分析，不阻塞队列。）
TARGETS: list[tuple[str, str, str, int]] = []
for bench in ["cifar100", "imagenetr"]:
    for m in ["seq", "ewc", "olora"]:
        for s in range(5):
            TARGETS.append((bench, m, "default", s))
    for s in range(5):
        TARGETS.append((bench, "folora_v2", "v2f_l300_k16", s))
# EWC 调优基线（仅 cifar100，λ=30）
for s in range(5):
    TARGETS.append(("cifar100", "ewc", "ewc_lam30", s))
# prompt 基线（l2p / coda，2 基准 × 3 seed）
for bench in ["cifar100", "imagenetr"]:
    for m in ["l2p", "coda"]:
        for s in range(3):
            TARGETS.append((bench, m, "default", s))


def result_path(bench: str, method: str, tag: str, seed: int) -> Path:
    return (Path("experiments") / bench / method / (tag or "default")
            / f"seed{seed}" / "results.json")


def is_finished(t: tuple[str, str, str, int]) -> bool:
    p = result_path(*t)
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("finished"))
    except (json.JSONDecodeError, OSError):
        return False


def remaining() -> list[tuple[str, str, str, int]]:
    return [t for t in TARGETS if not is_finished(t)]


def main() -> None:
    while True:
        rem = remaining()
        if not rem:
            print("SUPERVISOR: 目标实验集全部 finished，退出。", flush=True)
            return
        print(f"SUPERVISOR: 剩余 {len(rem)} 个实验未完成，运行 run_full_queue。", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "scripts.run_full_queue"])
        time.sleep(30)


if __name__ == "__main__":
    main()
