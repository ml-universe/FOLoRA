"""汇总所有实验结果，生成论文表格数字（均值 ± std，over seeds）。

用法：python -m scripts.aggregate [--benchmark cifar100]

读取 experiments/{benchmark}/{method}/{tag}/seed*/results.json，输出：
- 控制台打印的汇总表；
- reports/summary_{benchmark}.json 与 reports/summary_{benchmark}.md。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean_std(values):
    n = len(values)
    m = sum(values) / n
    s = (sum((v - m) ** 2 for v in values) / n) ** 0.5
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="cifar100")
    ap.add_argument("--out_dir", default="experiments")
    args = ap.parse_args()

    root = Path(args.out_dir)
    # (method, tag, protocol) -> {seed: results}。protocol 用 num_tasks/epochs/rank
    # 区分「主协议（20任务×5epoch×rank16）」与「冒烟（10任务×2epoch×rank8）」，
    # 避免把不同协议的数字混在一起（曾踩坑：ewc_ref 10任务 15.81 被误当成 20任务基线）。
    grouped = defaultdict(dict)
    for rpath in root.glob(f"{args.benchmark}/**/seed*/results.json"):
        d = json.loads(rpath.read_text(encoding="utf-8"))
        cfg = d["config"]
        proto = (cfg.get("num_tasks"), cfg.get("epochs"), cfg.get("lora_rank"))
        grouped[(cfg["method"], cfg.get("tag", ""), proto)][cfg["seed"]] = d

    rows = []
    for (method, tag, proto), seeds in sorted(grouped.items()):
        accs = [seeds[s]["final_acc_cil"] for s in sorted(seeds)]
        fgts = [seeds[s]["forgetting_cil"] for s in sorted(seeds)]
        incs = [seeds[s]["incremental_acc_cil"] for s in sorted(seeds)]
        m_acc, s_acc = mean_std(accs)
        m_fgt, s_fgt = mean_std(fgts)
        proto_s = f"{proto[0]}t/{proto[1]}e/r{proto[2]}"
        rows.append({
            "method": method, "tag": tag, "proto": proto_s,
            "n_seeds": len(seeds),
            "final_acc_cil": m_acc, "final_acc_cil_std": s_acc,
            "forgetting_cil": m_fgt, "forgetting_cil_std": s_fgt,
            "incremental_acc_cil": mean_std(incs)[0],
        })

    # 打印
    print(f"\n=== {args.benchmark} 汇总（均值 ± std over seeds）===")
    print(f"{'method':<10}{'tag':<14}{'proto':<14}{'avgACC':>14}{'FGT':>14}{'incACC':>10}")
    for r in rows:
        print(f"{r['method']:<10}{r['tag']:<14}{r['proto']:<14}"
              f"{r['final_acc_cil']*100:>10.2f}±{r['final_acc_cil_std']*100:.2f}"
              f"{r['forgetting_cil']*100:>10.2f}±{r['forgetting_cil_std']*100:.2f}"
              f"{r['incremental_acc_cil']*100:>10.2f}")

    # 写文件
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / f"summary_{args.benchmark}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# {args.benchmark} 实验结果汇总\n",
             "| method | tag | proto | avgACC(%) | FGT(%) | incACC(%) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['tag'] or '-'} | {r['proto']} | "
            f"{r['final_acc_cil']*100:.2f}±{r['final_acc_cil_std']*100:.2f} | "
            f"{r['forgetting_cil']*100:.2f}±{r['forgetting_cil_std']*100:.2f} | "
            f"{r['incremental_acc_cil']*100:.2f} |")
    (out / f"summary_{args.benchmark}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写入 reports/summary_{args.benchmark}.md 和 .json")


if __name__ == "__main__":
    main()
