"""EWC 调优基线：λ=30（扫描确认的 EWC 最优）补多 seed，给出公平的 EWC 均值±std。

背景：EWC λ 扫描（cifar100 seed0）显示 λ=30 是单峰最优（acc 8.65 vs λ=100 的 8.20）。
为在论文主表用「调优后的 EWC」作公平基线，这里把 λ=30 补到 5 seed（seed0 已由
扫描产出，补 seed1-4）。

用法：python -m scripts.run_ewc_tuned   （幂等 + --resume）
"""

import json
import subprocess
import sys
from pathlib import Path

SEEDS = [1, 2, 3, 4]


def run_dir_of(seed: int) -> Path:
    return Path("experiments") / "cifar100" / "ewc" / "ewc_lam30" / f"seed{seed}"


def is_finished(d: Path) -> bool:
    r = d / "results.json"
    if not r.exists():
        return False
    return bool(json.loads(r.read_text(encoding="utf-8")).get("finished"))


def main():
    for seed in SEEDS:
        d = run_dir_of(seed)
        if is_finished(d):
            print(f"[skip] {d}")
            continue
        cmd = [sys.executable, "-u", "-m", "scripts.run_single",
               "--benchmark", "cifar100", "--num_tasks", "20",
               "--method", "ewc", "--seed", str(seed),
               "--tag", "ewc_lam30", "--resume", "--ewc_lambda", "30"]
        print(f"[run] ewc_lam30 seed{seed}")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
