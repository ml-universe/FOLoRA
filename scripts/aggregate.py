"""汇总实验结果，生成论文表格数字（均值 ± std over seeds）。

**主指标是 NCM**（冻结特征的最近类均值），不是 head-CIL。
------------------------------------------------------------
这个项目里有两套数字，**键名却相同**，是本仓库最大的踩坑来源：

  experiments/**/results.json  的 `final_acc_cil`   -> 旧的 head-based CIL，约 0.07-0.11
  reports/ncm/**/*.json        的 `final_acc_cil`   -> NCM 协议，约 0.65-0.79

两者都叫 `final_acc_cil`，但差了 10 倍。早期就是因此把 10 任务/2 epoch/rank8 的
冒烟 run 当成主协议结果，得出了「λ 曲线无拐点、方法可能无效」的**假警报**。
所以本脚本：
  1. 只从 `reports/ncm/` 读主指标，绝不从 `results.json` 取精度；
  2. 用**每个 run 自己的 config.json** 校验 num_tasks/epochs/lora_rank，
     按协议分组，并在表里标注是否为主协议（20/5/16）；
  3. 旧的 head-CIL 仅作 `head_acc_LEGACY` 列保留，名字里带 LEGACY，防止误用。

用法：
  python -m scripts.aggregate                       # cifar100
  python -m scripts.aggregate --benchmark imagenetr
  python -m scripts.aggregate --main_protocol_only  # 只留 20/5/16（论文主表用这个）
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# 主协议（论文主表）：20 任务 × 5 epoch × rank16
MAIN_PROTOCOL = (20, 5, 16)


def mean_std(values):
    """样本标准差（ddof=1），与论文惯例和本仓库既有的 ±数值一致。

    早先用的是总体标准差（ddof=0），同一个 run 集合会得到不同的 ±
    （例：v2f_l300_k16 n=10 -> ddof=1 给 1.48，ddof=0 给 1.41），
    两套数字并存会让人以为数据变了。**统一用 ddof=1**。
    n=1 时无样本方差，def 返回 0（并应由调用方标注 n=1、不给显著性）。
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    return m, s


def run_config(run_dir: Path) -> dict:
    """读 run 自己的 config.json —— 协议校验的唯一可信来源。"""
    cp = run_dir / "config.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    # 退化：从 results.json 的 config 段取（老 run 可能没有独立 config.json）
    rj = run_dir / "results.json"
    if rj.exists():
        try:
            return json.loads(rj.read_text(encoding="utf-8")).get("config", {})
        except Exception:
            pass
    return {}


def norm_tag(tag) -> str:
    """把空 tag 与 'default' 归一。

    历史原因，**同一个配置**有的 run 记成 tag=""、有的记成 "default"
    （例如 ewc 的 10 个 λ=1000 run 被分成 5+5）。不归一就会把同一配置拆成两行、
    每行 n 只有一半，看起来像「有两个不同的 EWC 配置」，还会让 n 偏小、
    误判显著性。—— 这是纯记账问题，不影响任何 run 的结果。
    """
    t = (tag or "").strip()
    return t or "default"


def hyper_desc(method: str, cfg: dict) -> str:
    """把该 run 的关键超参写进表里，让每行自解释。

    没有这一列时，`ewc/default` 到底是不是 λ=1000 只能靠记忆和注释——
    而本项目的注释已被证明会过期。数字必须自带出处。
    """
    bits = []
    if method.startswith("folora"):
        bits.append(f"λ={cfg.get('folora_lambda')}")
        k = cfg.get("folora_topk") or cfg.get("lora_rank")
        bits.append(f"k={k}")
        if cfg.get("folora_weighted") is False:
            bits.append("等权")
    elif method == "ewc":
        bits.append(f"λ={cfg.get('ewc_lambda')}")
    elif method == "olora":
        agg = cfg.get("olora_aggregate", "sum")
        if agg != "sum":
            bits.append(f"agg={agg}")
    return " ".join(bits)


def main():
    ap = argparse.ArgumentParser(description="汇总实验（主指标 = NCM）")
    ap.add_argument("--benchmark", default="cifar100")
    ap.add_argument("--out_dir", default="experiments")
    ap.add_argument("--cache_root", default="reports/ncm")
    ap.add_argument("--main_protocol_only", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_dir)
    cache_root = Path(args.cache_root)

    # (method, tag, proto) -> {seed: {...}}
    grouped = defaultdict(dict)
    missing_ncm = []
    for rd in sorted(root.glob(f"{args.benchmark}/**/seed*/results.json")):
        run_dir = rd.parent
        cfg = run_config(run_dir)
        method = cfg.get("method") or run_dir.parent.parent.name
        tag = norm_tag(cfg.get("tag") or run_dir.parent.name)
        seed = cfg.get("seed", int(run_dir.name.replace("seed", "") or 0))
        proto = (cfg.get("num_tasks"), cfg.get("epochs"), cfg.get("lora_rank"))

        # NCM 是主指标的唯一来源
        cache = cache_root / args.benchmark / f"{method}__{tag}__seed{seed}.json"
        if not cache.exists():
            missing_ncm.append(f"{method}/{tag or 'default'}/seed{seed}")
            continue
        try:
            ncm = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            missing_ncm.append(f"{method}/{tag or 'default'}/seed{seed} (缓存损坏)")
            continue

        # 超参进 key：万一两个 run 用了同一个 tag 但 λ 不同（历史上发生过），
        # 只按 tag 归组会把它们悄悄平均成一个不存在的配置。
        hyper = hyper_desc(method, cfg)
        grouped[(method, tag, proto, hyper)][seed] = {
            "acc": ncm.get("final_acc_cil"),
            "fgt": ncm.get("forgetting_cil"),
            "inc": ncm.get("incremental_acc_cil"),
            "head": ncm.get("head_acc_cil"),        # 旧指标，仅留档
        }

    rows = []
    for (method, tag, proto, hyper), seeds in sorted(grouped.items(),
                                                     key=lambda kv: str(kv[0])):
        if args.main_protocol_only and proto != MAIN_PROTOCOL:
            continue
        seeds = {s: v for s, v in seeds.items() if v["acc"] is not None}
        if not seeds:
            continue
        accs = [seeds[s]["acc"] for s in sorted(seeds)]
        fgts = [seeds[s]["fgt"] for s in sorted(seeds) if seeds[s]["fgt"] is not None]
        incs = [seeds[s]["inc"] for s in sorted(seeds) if seeds[s]["inc"] is not None]
        heads = [seeds[s]["head"] for s in sorted(seeds) if seeds[s]["head"] is not None]
        m_acc, s_acc = mean_std(accs)
        m_fgt, s_fgt = mean_std(fgts)
        rows.append({
            "method": method, "tag": tag, "hyper": hyper,
            "proto": f"{proto[0]}t/{proto[1]}e/r{proto[2]}",
            "is_main_protocol": proto == MAIN_PROTOCOL,
            "n_seeds": len(seeds), "seeds": sorted(seeds),
            "ncm_acc": m_acc, "ncm_acc_std": s_acc,
            "ncm_fgt": m_fgt, "ncm_fgt_std": s_fgt,
            "ncm_inc": mean_std(incs)[0] if incs else None,
            "head_acc_LEGACY": mean_std(heads)[0] if heads else None,
        })

    print(f"\n=== {args.benchmark} 汇总 · 主指标 = NCM（冻结特征最近类均值）===")
    print("    head_acc_LEGACY 列是旧 head-CIL 指标，**不要用**，仅为此前误用留痕\n")
    print(f"{'method':<11}{'tag':<17}{'hyper':<14}{'proto':<13}{'n':>3}  "
          f"{'NCM avgACC':>16}{'NCM FGT':>16}{'head_LEGACY':>13}")
    for r in rows:
        flag = "" if r["is_main_protocol"] else "  <- 非主协议"
        print(f"{r['method']:<11}{r['tag']:<17}{r['hyper']:<14}{r['proto']:<13}{r['n_seeds']:>3}  "
              f"{r['ncm_acc']*100:>9.2f}±{r['ncm_acc_std']*100:<5.2f}"
              f"{r['ncm_fgt']*100:>9.2f}±{r['ncm_fgt_std']*100:<5.2f}"
              f"{(r['head_acc_LEGACY'] or 0)*100:>12.2f}{flag}")
    if missing_ncm:
        print(f"\n注意：{len(missing_ncm)} 个 run 还没跑 NCM 评估，已排除（跑 "
              f"`python -m scripts.eval_ncm_sweep` 补上）：")
        for x in missing_ncm[:12]:
            print(f"   {x}")
        if len(missing_ncm) > 12:
            print(f"   ... 另外 {len(missing_ncm) - 12} 个")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / f"summary_{args.benchmark}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# {args.benchmark} 实验结果汇总（主指标 = NCM）\n",
             "> `head_acc_LEGACY` 是旧的 head-CIL 指标（约 0.08），与本表其余列不可比，"
             "保留仅为追溯此前误用。\n",
             "| method | tag | hyper | proto | n | NCM avgACC(%) | NCM FGT(%) | head_LEGACY(%) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = "" if r["is_main_protocol"] else " ⚠"
        lines.append(
            f"| {r['method']} | {r['tag']} | {r['hyper'] or '-'} | {r['proto']}{flag} | {r['n_seeds']} | "
            f"{r['ncm_acc']*100:.2f}±{r['ncm_acc_std']*100:.2f} | "
            f"{r['ncm_fgt']*100:.2f}±{r['ncm_fgt_std']*100:.2f} | "
            f"{(r['head_acc_LEGACY'] or 0)*100:.2f} |")
    (out / f"summary_{args.benchmark}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写入 reports/summary_{args.benchmark}.md 和 .json")


if __name__ == "__main__":
    main()
