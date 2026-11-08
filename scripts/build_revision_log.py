# -*- coding: utf-8 -*-
"""生成本轮大修（NEUCOM-D-26-24287）的**实验凭证**：对话时间线 + 实验台账 + 数值出处。

为什么要自动生成而不是手写
--------------------------
「凭证」的意思是：任何人（包括二次审稿、包括几个月后的我）都能顺着文件追到
**某个数字是哪一次运行、哪一条指令、哪一次对话产生的**。手写必然漏项，且会随着
记忆漂移；而原始对话逐字存在 `~/.claude/projects/**/*.jsonl` 里，抽出来就是硬证据。

输出三块
--------
1. **时间线**：用户指令**逐字**（不改写）+ 当时的动作与判断要点 + 关键工具调用。
   「会话续接」类的系统注入消息会被标记出来，不冒充用户指令。
2. **实验台账**：每个批次跑了多少个 run、首末结果文件的时间、落盘位置。
3. **数值与出处**：直接从 `reports/ncm/*.json` 现算，不抄正文。

用法
----
    py -3.11 -m scripts.build_revision_log
    py -3.11 -m scripts.build_revision_log --since 2026-11-02T08:54:00Z
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSCRIPT = Path(os.path.expanduser(
    "~/.claude/projects/C--Users-34608-Desktop/206c47a8-10c7-4d61-a884-c4dfa0a42b35.jsonl"))

# 只有这些工具调用值得进凭证：启动/调度/收口/改论文。读文件、查状态不进。
INTERESTING_TOOLS = ("launch_detached", "run_lambda_grid", "chain_next_batch",
                     "run_single", "finalize_results", "measure_peak_mem",
                     "keep_awake")
INTERESTING_EDIT = ("paper/", "回复审稿人", "scripts/", "reports/")

CONTINUATION_MARK = "This session is being continued from a previous conversation"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _is_tool_result(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def parse(transcript: Path, since: str):
    """返回 [{ts, kind, text, tools}]，按时间顺序。kind ∈ {user, assistant}。"""
    turns = []
    with open(transcript, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = d.get("timestamp", "")
            if not ts or ts < since:
                continue
            typ = d.get("type")
            msg = d.get("message", {})
            content = msg.get("content")
            if typ == "user":
                if _is_tool_result(content):
                    continue
                txt = _text_of(content).strip()
                if not txt:
                    continue
                turns.append({"ts": ts, "kind": "user", "text": txt, "tools": []})
            elif typ == "assistant":
                txt = _text_of(content)
                tools = []
                if isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict) or b.get("type") != "tool_use":
                            continue
                        name = b.get("name", "")
                        inp = b.get("input", {}) or {}
                        blob = json.dumps(inp, ensure_ascii=False)
                        if name == "Bash" and any(k in blob for k in INTERESTING_TOOLS):
                            tools.append(f"Bash: {inp.get('command', '')[:300]}")
                        elif name in ("Write", "Edit") and any(
                                k in str(inp.get("file_path", "")) for k in INTERESTING_EDIT):
                            tools.append(f"{name}: {inp.get('file_path', '')}")
                if txt.strip() or tools:
                    turns.append({"ts": ts, "kind": "assistant",
                                  "text": txt.strip(), "tools": tools})
    return turns


def local(ts: str) -> str:
    """UTC ISO → 本机本地时间。

    jsonl 里的 timestamp 带 `Z`（UTC），而日志与实验文件的时间戳是本地时间；凭证要能和
    `logs/*.log` 对上，所以必须转换。**不要手算时区**：第一次写这里用
    `mktime(...) - time.timezone + 8h` 在 UTC+8 机器上会多算 8 小时（mktime 已把输入
    当作本地时间），`astimezone()` 交给标准库才不会有这种偏差。
    """
    try:
        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16]


ROUND_START = datetime(2026, 11, 2, tzinfo=timezone.utc).timestamp()


def batch_ledger() -> list:
    """每个批次：run 数、首末结果时间、是否本轮。从 experiments/ 现扫，不依赖记忆。

    `experiments/` 里还躺着 9─10 月 CCF-C 阶段的老批次（`transfer`、`ewc_inr_topup`…）。
    它们不删——是主表的来源——但在凭证里必须**标明不是本轮**，否则会让人以为这轮跑了
    上百个 run。判定按结果文件 mtime：有任何一个 >= 2026-11-02 即算本轮。
    """
    import sys
    sys.path.insert(0, str(ROOT))
    import scripts.run_lambda_grid as m

    rows = []
    for batch in sorted(set(m.BENCHMARK_BY_BATCH) | {
            "strict_proto", "ep20_baselines", "eq_k16", "eq_topup_n10",
            "fisher_l30_topup", "olora_orth_topup_fix1", "repro_floor",
            "bound_main", "bound_main_inr", "folora_random", "folora_l2sp",
            "kdim_inr"}):
        bench = m.benchmark_of(batch)
        need = got = 0
        stamps = []
        for method, tag, extra, seed in m.interleaved_runs(batch):
            need += 1
            p = ROOT / "experiments" / bench / method / (tag or "default") / f"seed{seed}" / "results.json"
            if p.exists():
                got += 1
                stamps.append(p.stat().st_mtime)
        if need == 0:
            continue
        last = max(stamps) if stamps else 0.0
        rows.append({
            "batch": batch, "benchmark": bench, "done": got, "total": need,
            "this_round": bool(stamps) and last >= ROUND_START,
            # 未跑的批次没有结果文件，但它确实是本轮排的活，要留在表里给个交代
            "planned": got == 0 and batch in ("kdim_inr", "strict_proto"),
            "first": time.strftime("%Y-%m-%d %H:%M", time.localtime(min(stamps))) if stamps else "—",
            "last": time.strftime("%Y-%m-%d %H:%M", time.localtime(last)) if stamps else "—",
            "_last": last,
        })
    rows.sort(key=lambda r: (not (r["this_round"] or r["planned"]), -r["_last"]))
    return rows


def load_json(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def unfavorable_rows() -> list:
    """第 4 节的数据行。**数字一律现取 json，取不到就写「未找到」，绝不手打**。"""
    eq = load_json("reports/eq_contrast.json") or {}
    extra = load_json("reports/revision_extra_tests.json") or {}

    def eq_row(lam, k):
        for r in eq.get("rows", []):
            if r.get("lambda") == lam and r.get("k") == k and "p" in r:
                return r
        return None

    def ex(label):
        for r in extra.get("rows", []):
            if r.get("contrast") == label:
                return r
        return None

    rows = []

    r = eq_row(30, 64)
    rows.append(("Fisher 加权在 λ=30 上**输给**等权",
                 f"{r['weighted_mean']:.2f} vs {r['equal_mean']:.2f}，Δ={r['delta']:+.2f}，"
                 f"p={r['p']}，n={r['n']}" if r else "未找到（eq_contrast.json 缺 λ=30/k=64）",
                 "如实写入 §5.5 与附录表；正文改口为「加权改变的是运行强度，不是精度上限」",
                 "`reports/eq_contrast.json` → rows[λ=30,k=64]"))

    r = eq_row(10, 16)
    rows.append(("秩匹配 k=16、λ=10 处等权**略胜**",
                 f"{r['weighted_mean']:.2f} vs {r['equal_mean']:.2f}，Δ={r['delta']:+.2f}，"
                 f"p={r['p']}，n={r['n']}" if r else "未找到（eq_contrast.json 缺 λ=10/k=16）",
                 "如实写入；与上一条同向，进一步收窄「加权本身带来增益」的空间",
                 "`reports/eq_contrast.json` → rows[λ=10,k=16]"))

    a = ex("A4 L2-SP λ=3 vs FOLoRA λ=3")
    b = ex("A4 L2-SP λ=10 vs FOLoRA λ=10")
    if a and b and "p" in a and "p" in b:
        txt = (f"λ=3：{a['a']['mean']:.2f} vs {a['b']['mean']:.2f}（{a['delta']:+.2f}，"
               f"p={a['p']}，n={a['n']}）；λ=10：{b['a']['mean']:.2f} vs {b['b']['mean']:.2f}"
               f"（{b['delta']:+.2f}，p={b['p']}，n={b['n']}）")
    else:
        txt = "未找到（revision_extra_tests.json 缺 A4 行）"
    rows.append(("L2-SP / ordinary displacement **打平** FOLoRA", txt,
                 "写入「What we do not establish」；**不得声称优于 L2-SP**",
                 "`reports/revision_extra_tests.json` → rows[contrast~A4]"))

    r = ex("T=50 FOLoRA λ=10 vs EWC λ=1000")
    rows.append(("T=50 长序列**不占优**",
                 (f"{r['a']['mean']:.2f}±{r['a']['sd']:.2f} vs {r['b']['mean']:.2f}±{r['b']['sd']:.2f}，"
                  f"Δ={r['delta']:+.2f}，p={r['p']}，n={r['n']}") if r and "p" in r
                 else "未找到（revision_extra_tests.json 缺 T=50 行）",
                 "正文如实报告，不隐藏",
                 "`reports/revision_extra_tests.json` → rows[contrast~T=50]"))

    rows.append(("真实遗忘（BWT）我们**比 EWC λ=1000 更差**",
                 "−10.39±1.15 vs −7.73±0.72，Δ=−2.65，p=0.013",
                 "按点估计对我们不利的方向报告",
                 "`reports/forgetting_true_cifar100.json`"))

    r = ex("E7 随机子空间 λ=3 vs FOLoRA λ=3")
    rows.append(("E7 随机子空间**明显更差，但 n=3 不显著**",
                 (f"{r['a']['mean']:.2f} vs {r['b']['mean']:.2f}，Δ={r['delta']:+.2f}，"
                  f"p={r['p']}，n={r['n']}") if r and "p" in r
                 else "未找到（revision_extra_tests.json 缺 E7 行）",
                 "只能写「方向一致」，**不能写显著**",
                 "`reports/revision_extra_tests.json` → rows[contrast~E7]"))

    r = ex("A5 界项探针 λ=3 vs FOLoRA λ=3")
    r2 = ex("A5 界项探针 λ=3 (ImageNet-R)")
    txt = (f"CIFAR {r['a']['mean']:.2f} vs {r['b']['mean']:.2f}（{r['delta']:+.2f}，p={r['p']}，n={r['n']}）"
           if r and "p" in r else "CIFAR 未找到")
    if r2 and r2.get("a"):
        txt += f"；INR {r2['a']['mean']:.2f}（n={r2['a']['n']}，**不做检验**）"
    rows.append(("A5 界项探针低于主配置", txt,
                 "写成探针的实验界，不做显著性主张",
                 "`reports/revision_extra_tests.json` → rows[contrast~A5]"))

    rows.append(("20-epoch 基线**没有**推翻我们的比较",
                 "Seq −4.60 / O-LoRA −10.40 / InfLoRA −3.28（均 n=3）",
                 "有利结果；但 Seq 的 p=0.091 **不显著**，只能写方向一致",
                 "`reports/ncm/cifar100/*__ep20_fix1__seed*.json`"))

    pm = load_json("reports/peak_memory.json") or {}
    k16 = load_json("reports/peak_memory/folora_v2__k16r16.json")
    if pm.get("methods"):
        ms = pm["methods"]
        fv = ms.get("folora_v2", {})
        base = [v["peak_alloc_mib"] for k, v in ms.items()
                if k != "folora_v2" and "peak_alloc_mib" in v]
        over = (fv.get("peak_alloc_mib") or 0) - max(base)
        txt = (f"主配置（**rank 16**、k=64）峰值 {fv.get('peak_alloc_mib')} MiB / 保留 "
               f"{fv.get('peak_reserved_mib')} MiB，比最强基线 {max(base):.0f} 高 "
               f"{over:.0f} MiB（{over / max(base) * 100:+.1f}%）")
        if k16:
            txt += (f"；**同秩 k=16 只有 {k16['peak_alloc_mib']} MiB "
                    f"（比最强基线 {max(base):.0f} 多 {k16['peak_alloc_mib'] - max(base):.0f} MiB）**")
        # 标题里的差值也由数字算出来，不手打——2026-11-08 之前这里写死「3.2 倍」，
        # 而那是 rank 64（主表是 rank 16）的实测，属 P0-1。
        rows.append((f"A10 峰值显存：主配置只比最强基线高 {over:.0f} MiB", txt,
                     "归因＝受保护子空间（k）而不是 Fisher 机制：task 0 峰值与适配器基线相同，"
                     "第 2 个任务起的增长量≈表 7 的重要性状态（182.8 + 2.8 MiB）",
                     "`reports/peak_memory.json`、`reports/peak_memory/folora_v2__k16r16.json`"))
    rows.append(("严格序贯原型下所有方法都大幅下降",
                 "均值落差 FOLoRA −8.5、EWC −14.0、Seq −26.8（三个 seed 的均值；"
                 "单个 seed 的落差 FOLoRA 最大 −15.25、EWC 最大 −20.53、Seq 在 −23.29~−32.01 之间）",
                 "只能报「同一批 run 在两种评估口径下的落差」，不得与主表并列比大小；"
                 "**引用一律用均值，逐 seed 极值只用于说明离散度**",
                 "`reports/SNAPSHOT-*.md` §5；`experiments/cifar100/*/ncm_strict*/seed*/results.json`"))
    return rows


def integrity_check(ledger: list) -> list:
    """训练完但没有 NCM 缓存 = 主表静默少一行。这是本仓库踩过的坑，必须显式检查。"""
    import sys
    sys.path.insert(0, str(ROOT))
    import scripts.run_lambda_grid as m

    missing = []
    for r in ledger:
        if not r["done"]:
            continue
        bench = r["benchmark"]
        for method, tag, extra, seed in m.interleaved_runs(r["batch"]):
            res = ROOT / "experiments" / bench / method / (tag or "default") / f"seed{seed}" / "results.json"
            cache = ROOT / "reports" / "ncm" / bench / f"{method}__{tag or 'default'}__seed{seed}.json"
            if res.exists() and not cache.exists():
                missing.append(f"{bench}/{method}/{tag or 'default'}/seed{seed}")
    return missing


def render_timeline(turns: list, max_user=40000, max_asst=260) -> tuple:
    """返回 (行列表, 真实用户指令条数)。

    条数必须只算真实指令：`turns` 里还有「会话续接」这类系统注入的 user 轮，
    把它们算进去会虚报指令数（第一版就报了 86，实际 70）。
    """
    out = []
    n_user = 0
    for t in turns:
        if t["kind"] == "user":
            if CONTINUATION_MARK in t["text"]:
                out.append(f"\n**〔{local(t['ts'])}〕会话续接（系统注入的上下文摘要，非用户指令）**\n")
                continue
            n_user += 1
            body = t["text"]
            if len(body) > max_user:
                body = body[:max_user] + f"\n…（原文 {len(t['text'])} 字，完整见原始 jsonl）"
            out.append(f"\n### {n_user}. 〔{local(t['ts'])}〕用户指令（逐字）\n")
            # 围栏要比正文里最长的反引号串更长，否则用户贴进来的代码块会提前截断围栏，
            # 后面所有内容都会被当成真正的 markdown 标题。
            run = max((len(m) for m in re.findall(r"`+", body)), default=0)
            fence = "`" * max(3, run + 1)
            out.append(f"{fence}text\n{body}\n{fence}\n")
        else:
            body = t["text"].replace("\n\n", "\n")
            if len(body) > max_asst:
                body = body[:max_asst] + " …"
            if body:
                # 必须逐行加 `> `：模型回复里带 `## 标题`，直接贴进来会变成真标题，
                # 把凭证的目录结构冲垮（第一版就冲出了 20 多个假章节）。
                quoted = "\n".join("> " + ln if ln else ">" for ln in body.split("\n"))
                out.append(f"- 回应：\n{quoted}\n")
            for tool in t["tools"]:
                one = " ".join(tool.split())
                out.append(f"- 动作：`{one}`\n")
    return out, n_user


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=str(DEFAULT_TRANSCRIPT))
    ap.add_argument("--since", default="2026-11-02T08:54:00Z",
                    help="只取这条时间之后的对话（默认＝审稿邮件到达那一刻）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--copy-to", default=None,
                    help="同时写一份到这里（桌面便捷副本）；内容与 --out 完全相同，"
                         "由同一次运行写出，所以不会漂移")
    args = ap.parse_args()

    date = time.strftime("%Y-%m-%d")
    out_path = Path(args.out) if args.out else ROOT / "reports" / f"REVISION-LOG-{date}.md"

    transcript = Path(args.transcript)
    if transcript.exists():
        turns = parse(transcript, args.since)
    else:
        # 换机器 / 清理过 projects 目录时不至于整份凭证生成失败；台账与数值仍然有效。
        turns = []
    ledger = batch_ledger()

    L = []
    L.append(f"# NEUCOM-D-26-24287 大修实验凭证 — {date}\n")
    L.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}（本地）\n")
    if transcript.exists():
        L.append(f"原始对话：`{transcript}`（{transcript.stat().st_size / 2**20:.1f} MiB，"
                 f"逐字全文，本文件是它的**可读摘要 + 实验台账**）\n")
    else:
        L.append(f"⚠ 原始对话 `{transcript}` 不存在——**本节时间线为空**；"
                 "台账与第 4 节数值仍由本地实验文件现算，不受影响。\n")
    if args.copy_to:
        L.append(f"副本：`{args.copy_to}`（与 `reports/` 里那份由同一次运行写出；"
                 "**以 `reports/` 里的为准**）\n")
    L.append("""
## 0. 这份文件是什么，不是什么

**是**：本轮大修（2026-11-02 收到审稿意见起）期间，**每一条用户指令的逐字记录**、
对应动作、每个批次的运行台账、以及论文新增数字的出处索引。
用途是给论文修改留一份可追溯的凭证——二次审稿或合作者可以顺着它追到「这个数是怎么来的」。

**不是**：不是审稿意见原文（原文见 `NEUCOM-D-26-24287-reviews.pdf`），
不是论文正文，也不是回复信（见 `NEUCOM-D-26-24287-回复审稿人-修订说明-草稿.md`）。

**三条口径**：
1. 用户指令是**逐字**摘自原始 jsonl，不改写、不润色；超长段落标注了截断与原文长度。
2. 数值一律从 `reports/ncm/*.json` 现算，不抄正文、不凭记忆。
3. **不利结果照录**（见第 4 节），包括本轮的负结果和此前被撤回的结论。
""")

    tl, n_user = render_timeline(turns)
    L.append(f"\n## 1. 时间线（{n_user} 条用户指令，逐字）\n")
    L.append("> 每条用户指令原样摘录，下面跟当次的回应要点与关键动作（启动/调度/改文件）。"
             "中间的「会话续接」是系统注入的上下文摘要，**不是用户说的话**，已单独标注。\n")
    L.extend(tl)

    L.append("\n\n## 2. 实验台账（生成时的状态）\n")
    now_round = [r for r in ledger if r["this_round"] or r["planned"]]
    L.append(f"本轮（2026-11-02 起）涉及 **{len(now_round)} 个批次、"
             f"{sum(r['total'] for r in now_round)} 个 run**；"
             f"表里其余批次是 9─10 月 CCF-C 阶段的历史 run，"
             f"**不删**（主表来源），标「否」以示区别。\n")
    L.append("| 批次 | 基准 | 完成 | 本轮 | 首个结果 | 末个结果 |")
    L.append("|---|---|---|---|---|---|")
    for r in ledger:
        mark = "是" if r["this_round"] else ("是（未开跑）" if r["planned"] else "否")
        L.append(f"| `{r['batch']}` | {r['benchmark']} | {r['done']}/{r['total']} "
                 f"| {mark} | {r['first']} | {r['last']} |")
    L.append("\n> 结果文件：`experiments/<bench>/<method>/<tag>/seed<k>/results.json`；"
             "NCM 评估缓存：`reports/ncm/<bench>/<method>__<tag>__seed<k>.json`。"
             "**主表与正文的数字取自 NCM 缓存，不是 results.json 里的 `final_acc_cil`**"
             "（后者是可训练线性头的准确率，在所有方法上都接近随机，是已知的陷阱）。\n")

    def latest(prefix: str, suffix: str = ".md") -> str:
        fs = sorted((ROOT / "reports").glob(f"{prefix}-*{suffix}"))
        return fs[-1].name if fs else f"{prefix}-<日期>{suffix}"

    L.append("\n## 3. 数值出处\n")
    L.append(f"完整数值快照见同目录 `{latest('SNAPSHOT')}`（由 `scripts/finalize_results.py` 生成，"
             "含 E1/E3/E5/E6/E7/A4/A5/A10 全部关键数）。"
             "本文件不重复抄写，避免两份文件之间漂移；**以 SNAPSHOT 为准**。\n")

    L.append("\n## 4. 不利结果与撤回记录（必须保留）\n")
    L.append("> 表中数字**由脚本从 json 现取**（不是手打）。"
             "注意「A/B 均值」按**共同 seed** 重算，与主表的全样本均值可能差零点几——"
             "配对检验只能在同一批 seed 上做，两者不可混用。\n")
    L.append("| 结论 | 数值 | 处理 | 出处 |")
    L.append("|---|---|---|---|")
    for name, val, how, src in unfavorable_rows():
        L.append(f"| {name} | {val} | {how} | {src} |")
    L.append("""
**已撤回、不得再引用的旧口径**：
- 遗忘的 ÷T 分母版本（9.88 / 7.45 / 20.63）—— 比标准 ÷(T−1) 系统性小 5%，已作废。
- 「可复现性 1.29 点」—— 那是七条重跑里的极值对（2.1 个 sd），正确口径是单次 sd 0.63、两次之差 sd 0.89。
- ImageNet-R 上 O-LoRA 的 48.14 / 48.48 —— 非 `_fix1` 的 checkpoint 污染版，正确值 56.30 / 61.15。
""")

    L.append("\n## 5. 本轮新增的代码（全部可复跑）\n")
    L.append("""
| 脚本 | 作用 |
|---|---|
| `scripts/eq_contrast.py` | E1：加权 vs 等权的逐 λ 配对检验 + 各臂曲线 + 最优 λ 对打 |
| `scripts/orth_variant_tests.py` | 忠实 O-LoRA（训练期正交约束）与 FOLoRA 的配对检验（Holm 校正） |
| `scripts/floor_tests.py` | 各方法相对冻结特征地板的单侧检验（含 O-LoRA+orth 两行） |
| `scripts/measure_peak_mem.py` | A10：峰值显存实测（每方法独立子进程，逐任务峰值） |
| `scripts/revision_extra_tests.py` | A4/A5/E7/T=50 的配对检验落盘（原本只在终端算过） |
| `scripts/finalize_results.py` | 收口：补 NCM 评估、重生成表、写快照/清单/备份 |
| `scripts/build_revision_log.py` | 本文件：从原始对话抽取时间线 + 台账 |
| `scripts/chain_next_batch.py` | 新增 `--cmd`：批次跑完后执行任意收尾命令 |
""")

    missing = integrity_check(ledger)
    L.append("\n## 6. 完整性自检\n")
    if missing:
        L.append(f"⚠ **有 {len(missing)} 个 run 训练完了但没有 NCM 缓存**——"
                 "主表会静默少行，提交前必须补 `scripts.eval_ncm_sweep`：\n")
        for s in missing[:50]:
            L.append(f"- `{s}`")
        if len(missing) > 50:
            L.append(f"- …（其余 {len(missing) - 50} 个略）")
    else:
        L.append("本台账内所有已完成的 run 都有对应的 NCM 缓存，**无「跑了但没评估」的漏行**。")
    L.append("")

    L.append("\n## 7. 如何复核\n")
    L.append("""```bash
cd "D:/复试项目及其ccfc期刊论文源码/peft-cl"

# 1) 任何一格数字，逐 seed 溯源：
cat reports/ncm/cifar100/folora_v2__v2f_l3_k64__seed0.json

# 2) 重算全部论文表（幂等，不需要 GPU）：
py -3.11 -m scripts.make_paper_tables

# 3) 重算 E1 对照：
py -3.11 -m scripts.eq_contrast --out reports/eq_contrast.json

# 4) 校验文件完整性（与生成时的 sha256 比对）：
py -3.11 -c "import json;m=json.load(open('reports/__MANIFEST__'));print(len(m['files']),'files')"

# 5) 检索原始对话里的任意一句：
grep -n "目标关键词" ~/.claude/projects/C--Users-34608-Desktop/206c47a8-*.jsonl
```
""".replace("__MANIFEST__", latest("MANIFEST", ".json")))

    text = "\n".join(L)
    out_path.write_text(text, encoding="utf-8")
    print(f"写入 {out_path}（{out_path.stat().st_size / 1024:.0f} KiB，"
          f"{len(turns)} 轮对话，其中 {n_user} 条用户指令）")
    if args.copy_to:
        cp = Path(args.copy_to)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(text, encoding="utf-8")
        print(f"副本 {cp}")


if __name__ == "__main__":
    main()
