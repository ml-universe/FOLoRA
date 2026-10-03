"""重跑受 P0-1 影响的 run，写入**新标签** `<tag>_fix1`。

背景（P0-1）
------------
`multi_lora.py` 的 O-LoRA / InfLoRA 曾把逐任务 adapter 只存最后一个（state_dict 里
`.adapters.<i>.` 只覆盖 `i=19`）。于是 NCM 评估读到的「模型」并不是训练产出的模型：
48x 个 key -> 26 个 key，缺失的 18 个 adapter 在加载时静默跳过。修复已落地（存满
`0..19`），但**旧 checkpoint 是被污染的证据**，不能用，必须按原超参重跑。

纪律（不可违背）
----------------
1. **绝不覆盖旧 run 目录**。新标签一律加 `_fix1` 后缀，旧目录原地保留 —— 它们是
   "污染前"的唯一物理证据，且 [[folora-data-integrity-rule]] 要求不利结果只改讲法、
   不删不改。
2. **超参只能来自原 run 的 `config.json`**，不许手抄。除 `tag` 外任何字段都不改。
3. 回放必须**无损**：凡 `CLConfig` 里 CLI 未暴露的字段，其盘上取值必须等于
   `CLConfig` 默认值；否则本脚本直接报错退出（说明 CLI 覆盖不全，回放会偏）。
4. 幂等：新 run 的 `results.json` 若 `finished`，跳过。中断后可重跑本脚本续上。

用法
----
    python -m scripts.retrain_fix1                 # 跑全部目标
    python -m scripts.retrain_fix1 --dry-run       # 只打印将要执行的命令
    python -m scripts.retrain_fix1 --only bench:method:tag
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

from peft_cl.utils.config import CLConfig

# 论文正文实际引用的 tag（其余 tag 只在 `%` 注释里出现，不重训）。
# olora_orth_l0.1 / olora_orth_l1 同时受 §5.4 引用（05_experiments.tex:653 明写
# 68.44±0.50 / 48.14-48.48, n=3 each），**两个 λ 都要**。
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

# run_single.py 暴露成 CLI 的 config 字段；其余字段必须等于默认值（见 check_replayable）。
CLI_EXPOSED = {
    "benchmark", "num_tasks", "method", "seed", "epochs", "batch_size", "lr",
    "lora_rank", "lora_alpha", "folora_lambda", "folora_topk", "folora_weighted",
    "olora_aggregate", "olora_orth_lambda", "ewc_lambda", "prompt_pool_size",
    "prompt_length", "prompt_topk", "fisher_batches", "data_root", "out_dir",
    "bound_probe",
}
DEFAULTS = {f.name: f.default for f in dataclasses.fields(CLConfig)}


def fix1_tag(tag: str) -> str:
    return f"{tag}_fix1"


def check_replayable(cfg: dict, where: str) -> None:
    """凡 CLI 未暴露的字段，盘上值必须等于默认值，否则回放会失真。"""
    bad = []
    for k, v in cfg.items():
        if k in CLI_EXPOSED or k in ("tag", "seed"):
            continue
        if k in DEFAULTS and v != DEFAULTS[k]:
            bad.append(f"{k}={v!r} (默认 {DEFAULTS[k]!r})")
    unknown = [k for k in cfg if k not in DEFAULTS and k not in CLI_EXPOSED]
    if bad or unknown:
        raise SystemExit(
            f"[回放不可信] {where}\n"
            f"  与默认值不同的未暴露字段: {bad or '无'}\n"
            f"  CLConfig 里不认识的字段: {unknown or '无'}\n"
            f"  => 必须先扩 run_single.py 的 CLI，否则重训结果不是原配置。"
        )


def build_cmd(cfg: dict, tag: str) -> list:
    cmd = [sys.executable, "-m", "scripts.run_single"]
    for k, v in sorted(cfg.items()):
        if k not in CLI_EXPOSED:
            continue
        if isinstance(v, bool):            # store_true 型
            if v:
                cmd.append(f"--{k}")
            continue
        cmd += [f"--{k}", str(int(v) if isinstance(v, bool) else v)]
    cmd += ["--tag", tag]
    return cmd


def run_dir_of(root: Path, bench: str, method: str, tag: str, seed: int) -> Path:
    return root / "experiments" / bench / method / tag / f"seed{seed}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", action="append", default=[],
                    help="限制为 bench:method:tag（可重复）")
    args = ap.parse_args()

    # 本脚本要挂数小时~数天，输出必须**逐行**可见：python 的 stdout 重定向到文件时
    # 默认是块缓冲（约 8KB），于是进度行会全部卡在缓冲区里，日志只剩 trainer 的
    # logger 输出，看起来像「卡住了」。2026-10-03 实际踩到过。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    root = Path(args.root).resolve()
    only = {tuple(s.split(":")) for s in args.only}

    queue = []
    for bench, method, tag in TARGETS:
        if only and (bench, method, tag) not in only:
            continue
        src_dir = root / "experiments" / bench / method / tag
        if not src_dir.exists():
            print(f"  · 跳过 {bench}/{method}/{tag}：源目录不存在")
            continue
        for seed_dir in sorted(src_dir.glob("seed*")):
            if not (seed_dir / "config.json").exists():
                continue
            seed = int(seed_dir.name[4:])
            cfg = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
            check_replayable(cfg, str(seed_dir))
            queue.append((bench, method, tag, seed, cfg))

    print(f"目标 run 共 {len(queue)} 个（源超参逐字段核对通过）\n")

    new_tag = {}
    done = skipped = failed = 0
    for bench, method, tag, seed, cfg in queue:
        nt = fix1_tag(tag)
        out = run_dir_of(root, bench, method, nt, seed)
        res = out / "results.json"
        if res.exists():
            try:
                if json.loads(res.read_text(encoding="utf-8")).get("finished"):
                    print(f"[skip] 已完成 {bench}/{method}/{nt}/seed{seed}")
                    skipped += 1
                    new_tag[(bench, method, tag)] = nt
                    continue
            except Exception:
                pass
        cmd = build_cmd(cfg, nt)
        print(f"[run ] {bench}/{method}/{tag}/seed{seed}  ->  {nt}")
        if args.dry_run:
            print("        " + " ".join(cmd))
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=root)
        dt = time.time() - t0
        ok = r.returncode == 0 and res.exists() and json.loads(
            res.read_text(encoding="utf-8")).get("finished")
        print(f"        {'✅' if ok else '❌'} {dt/60:.1f} min")
        if ok:
            done += 1
            new_tag[(bench, method, tag)] = nt
        else:
            failed += 1
            print(f"        ⚠ 失败（returncode={r.returncode}），继续下一个；"
                  f"修好后重跑本脚本即可续上。")

    print(f"\n完成 {done} / 跳过 {skipped} / 失败 {failed}")
    if not args.dry_run and new_tag:
        print("\n新标签（供 paper/scripts 引用）：")
        for (b, m, t), nt in sorted(new_tag.items()):
            print(f"   {b:<10} {m:<8} {t}  ->  {nt}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
