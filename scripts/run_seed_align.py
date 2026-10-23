"""把各基准的方法 seed 数对齐到 5（补 seed3/seed4），与 FOLoRA 的 5 seed 一致。

CIFAR-100：seq/ewc/olora 补 seed3/4（folora_v2 已有 5 seed）。
ImageNet-R：seq/ewc/olora/folora_v2 全部补 seed3/4。

主协议：20 任务 × 5 epoch × rank16；folora_v2 用 λ=300/k=16（tag=v2f_l300_k16）。
幂等 + --resume（同 run_imagenetr）。

用法（项目根目录）：python -m scripts.run_seed_align
"""

import json
import subprocess
import sys
from pathlib import Path

SEEDS = [3, 4]
FULL = ["--num_tasks", "20", "--epochs", "5", "--lora_rank", "16"]


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
            for m in ["seq", "ewc", "olora"]:
                runs.append((bench, m, "", seed, []))
            runs.append((bench, "folora_v2", "v2f_l300_k16", seed,
                         ["--folora_lambda", "300", "--folora_topk", "16"]))

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
        print(f"[run] {bench} {method} {tag or 'default'} seed{seed}")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
