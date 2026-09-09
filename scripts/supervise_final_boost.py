"""最终强化实验守护：循环 run_final_boost 直到 30 个目标实验全部 finished（断电自动续训）。

与 supervise_queue.py / supervise_pilot20.py 同构。外层反复调 run_final_boost
（内部每个 run_single 带 --resume、幂等），解决 OOM/休眠/关机导致的静默死。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_final_boost import TARGETS


def result_path(t) -> Path:
    bench, method, tag, seed, _epochs, _extra = t
    return Path("experiments") / bench / method / (tag or "default") / f"seed{seed}" / "results.json"


def is_finished(t) -> bool:
    p = result_path(t)
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("finished"))
    except (json.JSONDecodeError, OSError):
        return False


def remaining():
    return [t for t in TARGETS if not is_finished(t)]


def main() -> None:
    while True:
        rem = remaining()
        if not rem:
            print("SUPERVISOR: 全部目标 finished，退出。", flush=True)
            return
        print(f"SUPERVISOR: 剩余 {len(rem)} 个，运行 run_final_boost。", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "scripts.run_final_boost"])
        time.sleep(30)


if __name__ == "__main__":
    main()
