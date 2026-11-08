# -*- coding: utf-8 -*-
"""把这一轮补实验的**所有产物**收口、落盘、可核验（重启机器之前跑）。

为什么需要它
------------
2026-11-08 早上这一轮（E5 严格序贯原型 + R8-6 的 ImageNet-R k 格 + A10 峰值显存）
跑完之后用户要重启电脑。重启会杀掉所有进程，之后要凭 `reports/` 里的文件继续改论文。
所以收口必须做到三件事，缺一件都会让「重启后接着干」变成「重启后重新找数」：

1. **补评估**：训练完不等于有 NCM 数——`eval_ncm_sweep` 是幂等的缓存补算，
   任何一个 run 漏评估，主表就静默少一行（本仓库踩过：InfLoRA 差一点整行空着）。
2. **重算派生表**：`make_paper_tables` / `eq_contrast` / `orth_variant_tests` /
   `floor_tests` 全部由 `reports/ncm/*.json` 派生，重跑一次保证 .tex 与 json 同步。
   **这些脚本都是幂等的**，重跑不会改动已有结论。
3. **快照 + 清单 + 备份**：写一份人读的 `SNAPSHOT-<日期>.md`（所有关键数字和出处）、
   一份机读的 `MANIFEST-<日期>.json`（每个文件的 sha256，重启后可比对是否丢文件）、
   以及一个小 zip（只装 json，不装 37 GB 的 checkpoint）。

设计上的两条硬约束
------------------
* **不删任何东西**：本脚本只读实验结果、只写 `reports/`，不 `git clean`、不碰 `experiments/`。
* **任一步失败不中止**：评估失败只意味着少几个缓存，不该让快照与备份也做不成；
  每步的退出码都记进快照，谁失败一眼能看见。

用法
----
    py -3.11 -m scripts.finalize_results
    py -3.11 -m scripts.finalize_results --skip-eval     # GPU 被占时只做快照/备份
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# 控制台是 GBK：任何不可编码字符都会让 print 抛 UnicodeEncodeError，把收口
# 的最后一步退出码变成 1（全部产物其实已写完）。降级替换，不让它致命。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]

# 顺序即执行顺序；(显示名, 参数)
STEPS = [
    ("NCM 评估 cifar100", ["-m", "scripts.eval_ncm_sweep", "--benchmark", "cifar100",
                          "--num_workers", "1",
                          "--out", "reports/ncm_summary_cifar100.json"]),
    ("NCM 评估 imagenetr", ["-m", "scripts.eval_ncm_sweep", "--benchmark", "imagenetr",
                           "--num_workers", "1",
                           "--out", "reports/ncm_summary_imagenetr.json"]),
    ("重生成论文表", ["-m", "scripts.make_paper_tables"]),
    ("E1 加权 vs 等权", ["-m", "scripts.eq_contrast", "--out", "reports/eq_contrast.json"]),
    ("O-LoRA 正交变体检验", ["-m", "scripts.orth_variant_tests"]),
    ("地板检验", ["-m", "scripts.floor_tests"]),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_step(name: str, argv: list) -> dict:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-u", *argv], cwd=str(ROOT)).returncode
    dt = round(time.time() - t0, 1)
    print(f"[{name}] 退出码 {rc}，用时 {dt}s", flush=True)
    return {"step": name, "rc": rc, "seconds": dt}


def load_ncm(bench: str) -> dict:
    """{(method, tag): {seed: acc%}}，与主表/NCM 同口径。"""
    out = {}
    d = ROOT / "reports" / "ncm" / bench
    for p in sorted(d.glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            k = (j["method"], j["tag"])
            out.setdefault(k, {})[int(j["seed"])] = j["final_acc_cil"] * 100
        except Exception:
            continue
    return out


def stats(seedmap: dict):
    vals = list(seedmap.values())
    n = len(vals)
    if not n:
        return None
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return n, m, sd


def fmt(seedmap):
    s = stats(seedmap)
    if not s:
        return "—"
    n, m, sd = s
    return f"{m:.2f} ± {sd:.2f} (n={n})"


def build_snapshot(date: str, step_results: list) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# FOLoRA 补实验结果快照 — {date}", "",
             f"生成时间：{now}", "",
             "> 本文件由 `scripts/finalize_results.py` 生成。**数字全部来自 "
             "`reports/ncm/<bench>/*.json`**，不是从正文抄回来的；每个数都能用 "
             "`reports/ncm/<bench>/<method>__<tag>__seed<k>.json` 逐 seed 查到。",
             ""]

    lines += ["## 1. 本轮收口各步的退出码", "", "| 步骤 | 退出码 | 用时 (s) |", "|---|---|---|"]
    for r in step_results:
        lines.append(f"| {r['step']} | {r['rc']} | {r['seconds']} |")
    lines.append("")

    c100 = load_ncm("cifar100")
    inr = load_ncm("imagenetr")

    lines += ["## 2. E3 — 20 epoch 的非 Fisher 基线（R7-W4 / R8-8）", "",
              "同一批 run 在 5 epoch 与 20 epoch 下的对照。**预算更长，基线更差**。", "",
              "| 方法 | 5 epoch | 20 epoch |", "|---|---|---|"]
    for m, t5, t20 in [("seq", "default", "ep20_fix1"),
                       ("olora", "default_fix1", "ep20_fix1"),
                       ("inflora", "default_fix1", "ep20_fix1")]:
        lines.append(f"| {m} | {fmt(c100.get((m, t5), {}))} | {fmt(c100.get((m, t20), {}))} |")
    lines.append("")

    lines += ["## 3. E1 — Fisher 加权 vs 等权（R1-1 / R7-W3 / R8-2）", "",
              "| λ | 加权 k=64 | 等权 k=64 | 加权 k=16 | 等权 k=16 |", "|---|---|---|---|---|"]
    for lam in (1, 3, 10, 30):
        lines.append(f"| {lam} | {fmt(c100.get(('folora_v2', f'v2f_l{lam}_k64'), {}))} "
                     f"| {fmt(c100.get(('folora_v2', f'v2f_eq_l{lam}_k64'), {}))} "
                     f"| {fmt(c100.get(('folora_v2', f'v2f_l{lam}_k16'), {}))} "
                     f"| {fmt(c100.get(('folora_v2', f'v2f_eq_l{lam}_k16'), {}))} |")
    lines += ["", "配对检验见 `reports/eq_contrast.json`；最优 λ 对打见同文件的 `best_tuned`。", ""]

    lines += ["## 4. E6 / 变体对照", "", "| 行 | CIFAR-100 | ImageNet-R |", "|---|---|---|"]
    for label, tag, tag_inr in [
            ("FOLoRA 主配置 (λ=3,k=64)", "v2f_l3_k64", "v2f_l3_k64"),
            # **两个基准都用 `_fix1`**：非 fix1 的 INR run 是 adapter-checkpoint P0
            # 污染版（48.14/48.48），与论文正文的 56.30/61.15 差 8–13 分。
            # 2026-11-07 首次生成快照时这里错用了非 fix1，已修。
            ("O-LoRA 忠实版 (λ₁=0.1)", "olora_orth_l0.1_fix1", "olora_orth_l0.1_fix1"),
            ("O-LoRA 忠实版 (λ₁=1.0)", "olora_orth_l1_fix1", "olora_orth_l1_fix1"),
            ("E7 随机子空间 (λ=3)", "v2f_rand_l3_k64", None),
            ("A4 L2-SP (λ=3)", "v2f_l2sp_l3", None),
            ("A4 L2-SP (λ=10)", "v2f_l2sp_l10", None),
            ("A5 界项探针 (λ=3)", "bound_l3_k64", "bound_l3_k64")]:
        m = "olora" if label.startswith("O-LoRA") else "folora_v2"
        left = fmt(c100.get((m, tag), {}))
        right = fmt(inr.get((m, tag_inr), {})) if tag_inr else "—"
        lines.append(f"| {label} | {left} | {right} |")
    lines.append("")

    lines += ["## 5. E5 — 严格序贯原型（R7-W2）", "",
              "严格口径下原型在任务结束时冻结，评估只用终态特征；`标准 NCM` 是同一批 run "
              "按主表口径算出来的。**只能做差，不能与主表并列比大小**（换了评估规则）。", "",
              "| 方法 | seed | 严格协议 | 标准 NCM | 落差 |", "|---|---|---|---|---|"]
    for m, tag in [("seq", "ncm_strict"), ("ewc", "ncm_strict_l300"),
                   ("folora_v2", "ncm_strict_l3_k64")]:
        strict = {}
        for p in (ROOT / "experiments" / "cifar100").glob(f"{m}/{tag}/seed*/results.json"):
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                if j.get("finished") and j.get("strict_acc_cil") is not None:
                    strict[j["config"]["seed"]] = j["strict_acc_cil"] * 100
            except Exception:
                continue
        std = c100.get((m, tag), {})
        for s in sorted(strict):
            drop = f"{std[s] - strict[s]:+.2f}" if s in std else "—"
            stdtxt = f"{std[s]:.2f}" if s in std else "—"
            lines.append(f"| {m} | {s} | {strict[s]:.2f} | {stdtxt} | {drop} |")
    lines.append("")

    lines += ["## 6. A10 — 峰值显存实测", ""]
    pm = ROOT / "reports" / "peak_memory.json"
    if pm.exists():
        j = json.loads(pm.read_text(encoding="utf-8"))
        lines += ["| 方法 | 峰值分配 (MiB) | 峰值保留 (MiB) | 重要性估计 (s) |", "|---|---|---|---|"]
        for m, v in j.get("methods", {}).items():
            if "peak_alloc_mib" in v:
                lines.append(f"| {m} | {v['peak_alloc_mib']} | {v['peak_reserved_mib']} "
                             f"| {v['importance_seconds_total']} |")
            else:
                lines.append(f"| {m} | {v.get('status')} | — | — |")
        lines += ["", f"口径：{j.get('num_tasks')} 任务 × 每任务 {j.get('epochs_per_task')} epoch。"
                      "**峰值由模型 + 单 batch 决定，与 epoch 数无关**——这句话必须写进回复信。"]
        # 归因：默认 k=64 的峰值是基线约 3.2 倍，必须说清楚是「秩」而不是「Fisher 机制」。
        attr = ROOT / "reports" / "peak_memory" / "folora_v2__k16.json"
        if attr.exists():
            a = json.loads(attr.read_text(encoding="utf-8"))
            fv = j.get("methods", {}).get("folora_v2", {})
            base = [v["peak_alloc_mib"] for k, v in j.get("methods", {}).items()
                    if k != "folora_v2" and "peak_alloc_mib" in v]
            # 倍数由数字算出来，不手打：曾经写死「3.3 倍」，而表里实为 3.2×。
            ratio = (fv.get("peak_alloc_mib") or 0) / max(base)
            lines += ["", f"**归因（为什么 k=64 是 {ratio:.1f} 倍）**：同一条 FOLoRA，只把秩降到 16，"
                          "峰值回落到与基线同一水平 ⇒ 多出来的是**秩**的代价，不是 Fisher 机制。", "",
                      "| 配置 | 峰值分配 (MiB) | 峰值保留 (MiB) |", "|---|---|---|",
                      f"| FOLoRA k=64（默认） | {fv.get('peak_alloc_mib')} "
                      f"| {fv.get('peak_reserved_mib')} |",
                      f"| FOLoRA k=16（同秩对照，{a['num_tasks']} 任务） | {a['peak_alloc_mib']} "
                      f"| {a['peak_reserved_mib']} |",
                      f"| 四个基线的范围（rank=16） | {min(base):.1f}–{max(base):.1f} | — |",
                      "",
                      f"同秩下 FOLoRA 比最强的基线只多 **{a['peak_alloc_mib'] - max(base):.0f} MiB**"
                      f"（{(a['peak_alloc_mib'] / max(base) - 1) * 100:+.0f}%）。"
                      "task 0 的比较更干净（那时还没有受保护子空间，两者只差秩）："
                      f"{[t['peak_alloc_mib'] for t in json.loads((ROOT / 'reports' / 'peak_memory' / 'folora_v2.json').read_text(encoding='utf-8'))['per_task']][0]} "
                      f"vs {a['per_task'][0]['peak_alloc_mib']} MiB。"]
    else:
        lines.append("`reports/peak_memory.json` 不存在（A10 未完成或失败）。")
    lines.append("")

    lines += ["## 7. 文件清单", "",
              "见同目录 `MANIFEST-<日期>.json`（每个文件的 sha256、字节数、mtime）。"
              "备份 zip：`reports/_backup/FOLoRA-results-<日期>.zip`（只装 json，不含 checkpoint）。", ""]
    return "\n".join(lines)


def collect_files() -> list:
    files = []
    for p in sorted((ROOT / "reports").rglob("*.json")):
        if "_backup" in p.parts:
            continue
        # 排除历次 MANIFEST 自身：收集时它还是**上一版**，随后写新内容 ⇒ 清单里记的
        # sha 永远对不上，每次核验都报 1 条假阳性。新清单由 main() 显式加进 zip。
        if p.name.startswith("MANIFEST-"):
            continue
        files.append(p)
    for p in sorted((ROOT / "experiments").rglob("results.json")):
        files.append(p)
    for p in sorted((ROOT / "experiments").rglob("config.json")):
        files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="收口本轮补实验的全部产物")
    ap.add_argument("--skip-eval", action="store_true",
                    help="跳过会占 GPU 的评估步（只做派生表、快照、清单、备份）")
    ap.add_argument("--no-zip", action="store_true", help="不打包备份 zip")
    args = ap.parse_args()

    date = time.strftime("%Y-%m-%d")
    results = []
    for name, argv in STEPS:
        if args.skip_eval and name.startswith("NCM 评估"):
            print(f"[跳过] {name}", flush=True)
            results.append({"step": name, "rc": None, "seconds": 0})
            continue
        results.append(run_step(name, argv))

    # ---- 清单 ----
    files = collect_files()
    manifest = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "root": str(ROOT), "steps": results,
                "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                           "sha256": sha256(p)} for p in files]}
    mpath = ROOT / "reports" / f"MANIFEST-{date}.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(f["bytes"] for f in manifest["files"])
    print(f"\n清单：{mpath}（{len(files)} 个文件，共 {total / 2**20:.1f} MiB）", flush=True)

    # ---- 快照 ----
    snap = build_snapshot(date, results)
    spath = ROOT / "reports" / f"SNAPSHOT-{date}.md"
    spath.write_text(snap, encoding="utf-8")
    print(f"快照：{spath}", flush=True)

    # ---- 大修凭证（对话时间线 + 台账）----
    # 必须放在快照**之后**：凭证第 3 节指向最新 SNAPSHOT，先写它会指到上一份。
    lpath = ROOT / "reports" / f"REVISION-LOG-{date}.md"
    # 桌面那份是给用户直接打开看的便捷副本；由**同一次运行**写出，不会与 reports 版漂移。
    lcopy = Path("C:/Users/34608/Desktop/NEUCOM-D-26-24287-大修实验凭证.md")
    lrc = subprocess.run([sys.executable, "-u", "-m", "scripts.build_revision_log",
                          "--copy-to", str(lcopy)],
                         cwd=str(ROOT)).returncode
    if lpath.exists():
        print(f"凭证：{lpath}（rc={lrc}）", flush=True)
    else:
        print(f"⚠ 凭证未生成（rc={lrc}），手工跑：py -3.11 -m scripts.build_revision_log",
              flush=True)

    # ---- 备份 zip ----
    if not args.no_zip:
        zdir = ROOT / "reports" / "_backup"
        zdir.mkdir(exist_ok=True)
        zpath = zdir / f"FOLoRA-results-{date}.zip"
        # 按 arcname 去重：`collect_files()` 会把**上一轮**留下的 MANIFEST 也算进去，
        # 再显式加一次本轮的 MANIFEST 就会在同名条目上撞出
        # `UserWarning: Duplicate name`（第一次跑这里就撞了）。去重比 try/except 干净。
        entries = {str(p.relative_to(ROOT)): p for p in files}
        entries[str(mpath.relative_to(ROOT))] = mpath
        entries[str(spath.relative_to(ROOT))] = spath
        if lpath.exists():
            entries[str(lpath.relative_to(ROOT))] = lpath
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for arcname, p in entries.items():
                z.write(p, arcname=arcname)
        print(f"备份：{zpath}（{zpath.stat().st_size / 2**20:.1f} MiB，"
              f"{len(entries)} 个条目）", flush=True)

    bad = [r for r in results if r["rc"] not in (0, None)]
    if bad:
        # 不用 U+26A0：Windows 控制台是 GBK，该字符会让这行 print 自己抛
        # UnicodeEncodeError，把「哪几步失败」的信息一起吞掉。
        print(f"\n[警告] 有 {len(bad)} 步非 0 退出：" + ", ".join(f"{r['step']}(rc={r['rc']})"
                                                             for r in bad), flush=True)
        sys.exit(1)
    print("\n收口完成。", flush=True)


if __name__ == "__main__":
    main()
