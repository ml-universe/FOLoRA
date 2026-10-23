"""L2P / CODA-Prompt 基线实验（2 方法 × 2 基准 × 3 seed = 12 次）。

主协议：20 任务 × 5 epoch（与 LoRA 基线一致）。prompt 超参：
- L2P：pool=20, length=5, topk=5（Wang et al. 默认）
- CODA：components=100, length=8（Smith et al. 默认）

幂等 + --resume（同 run_imagenetr）。

用法（项目根目录）：python -m scripts.run_prompt_baselines
"""

import json
import subprocess
import sys
from pathlib import Path

SEEDS = [0, 1, 2]
FULL = ["--num_tasks", "20", "--epochs", "5"]

# (method, tag, extra args)
CONFIGS = [
    ("l2p", "", ["--prompt_pool_size", "20", "--prompt_length", "5",
                 "--prompt_topk", "5"]),
    ("coda", "", ["--prompt_pool_size", "100", "--prompt_length", "8"]),
]


def run_dir_of(bench: str, method: str, seed: int, tag: str) -> Path:
    return Path("experiments") / bench / method / (tag or "default") / f"seed{seed}"


def is_finished(d: Path) -> bool:
    r = d / "results.json"
    if not r.exists():
        return False
    return bool(json.loads(r.read_text(encoding="utf-8")).get("finished"))


def main():
    runs = []
    for seed in SEEDS:
        for bench in ["cifar100", "imagenetr"]:
            for method, tag, extra in CONFIGS:
                runs.append((bench, method, tag, seed, extra))

    for bench, method, tag, seed, extra in runs:
        d = run_dir_of(bench, method, seed, tag)
        if is_finished(d):
            print(f"[skip] {d}")
            continue
        cmd = [sys.executable, "-u", "-m", "scripts.run_single",
               "--benchmark", bench, "--method", method, "--seed", str(seed),
               "--resume"] + FULL + extra
        if tag:
            cmd += ["--tag", tag]
        print(f"[run] {bench} {method} seed{seed}")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
