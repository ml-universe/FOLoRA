"""EWC 调优基线：λ=30（扫描确认的 EWC 最优）补多 seed，给出公平的 EWC 均值±std。

!! STALE（2026-10-04 标注）：本文件头原先写的「λ=30 单峰最优（acc 8.65 vs λ=100 的
8.20）」是**旧的可训练头协议（head-CIL）**下的结论，已作废，**不要**据此在本协议下选
λ——那两个数本身（8 分档）就说明它是 head 口径。论文实际使用的 frozen-feature NCM
协议下，EWC 的调优最优是 λ=300（CIFAR-100，78.00）与 λ=1000（ImageNet-R，66.33）；
依据是 `reports/ncm_summary_{cifar100,imagenetr}.json` 与
`paper/tables/tab_ewc_lambda.tex`（由 `scripts/make_paper_tables.py` 生成）。
本脚本与它跑出的 λ=30 那批 run 保留作历史证据，**新的比较不要再跑它**。
原文（供溯源）：EWC λ 扫描（cifar100 seed0）显示 λ=30 是单峰最优（acc 8.65 vs λ=100
的 8.20）；为在论文主表用「调优后的 EWC」作公平基线，这里把 λ=30 补到 5 seed。

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
