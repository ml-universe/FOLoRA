"""看 `retrain_fix1.py` 的进度与 ETA。只读，随时可跑（不影响训练）。

用法：
    python -m scripts.retrain_status
    python -m scripts.retrain_status --log reports/retrain_fix1.log

判据说明：`retrain_fix1.py` 的 stdout 若被重定向，早期版本会块缓冲导致日志里看不到
`[run ]` 行；所以本脚本**不依赖日志**，而是直接扫 `experiments/`：以 `_fix1` 后缀的
run 目录为进度依据，用 `results.json` 的 `finished` 字段判完成。这样无论日志怎么缓冲
都能得到真实进度。
"""

import argparse
import json
import os
import time
from pathlib import Path

TARGETS = [
    ("cifar100", "olora", "default"),
    ("cifar100", "olora", "olora_orth_l0.1"),
    ("cifar100", "olora", "olora_orth_l1"),
    ("cifar100", "inflora", "default"),
    ("imagenetr", "olora", "default"),
    ("imagenetr", "olora", "olora_orth_l0.1"),
    ("imagenetr", "olora", "olora_orth_l1"),
    ("imagenetr", "inflora", "default"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--log", default="reports/retrain_fix1.log")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    total = done = running = 0
    durs = []            # 已完成 run 的耗时（分钟），用 config.json→results.json 的 mtime
    per_target = []
    now = time.time()

    for bench, method, tag in TARGETS:
        n_src = len(list((root / "experiments" / bench / method / tag).glob("seed*")))
        fin = 0
        cur = None
        cur_age = None
        for seed_dir in sorted((root / "experiments" / bench / method / tag).glob("seed*")):
            seed = seed_dir.name[4:]
            d = root / "experiments" / bench / method / f"{tag}_fix1" / f"seed{seed}"
            res, cfg = d / "results.json", d / "config.json"
            total += 1
            if res.exists():
                try:
                    if json.loads(res.read_text(encoding="utf-8")).get("finished"):
                        fin += 1
                        done += 1
                        # 耗时用「目录创建时刻 → results.json 落盘」。
                        # **不要**用 config.json 的 mtime：续跑会重写它，差值会缩到 0
                        # （2026-10-03 实测把中位耗时算成 0.0 min）。
                        # Windows 上 st_ctime 就是创建时间。
                        try:
                            durs.append((res.stat().st_mtime - os.path.getctime(d)) / 60)
                        except OSError:
                            pass
                        continue
                except Exception:
                    pass
            if cfg.exists():
                running += 1
                if cur_age is None or (now - cfg.stat().st_mtime) < cur_age:
                    cur = d.relative_to(root)
                    cur_age = now - cfg.stat().st_mtime
        per_target.append((bench, method, tag, fin, n_src, cur, cur_age))

    print(f"目标 {total} 个 run：完成 {done}，在跑 {running}\n")
    print(f"{'benchmark':<11}{'method':<9}{'tag':<20}{'完成':>6}{'/总数':>7}   当前")
    for bench, method, tag, fin, n, cur, age in per_target:
        c = f"  ← {cur}  ({age/60:.0f} min ago)" if cur else ""
        print(f"{bench:<11}{method:<9}{tag:<20}{fin:>6}{'/'+str(n):>7}{c}")

    if durs:
        med = sorted(durs)[len(durs) // 2]
        rem = total - done
        print(f"\n已完成 run 实测耗时：中位 {med:.1f} min（区间 {min(durs):.1f}–{max(durs):.1f}）")
        print(f"剩余 {rem} 个 → 预估还需 ≈ {med*rem/60:.1f} h")

    log = root / args.log
    if log.exists():
        tail = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        print(f"\n日志尾部（{log}）：")
        for l in tail[-4:]:
            print("   " + l.strip()[:110])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
