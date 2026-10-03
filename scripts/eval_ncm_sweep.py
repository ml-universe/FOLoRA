"""对全部已训练 run 做 NCM 重评估，聚合成标准协议主表（零训练、可断点续跑）。

背景（为什么要这么做）
----------------------
主表原先用「随任务增长的可训练线性头」做 CIL，实测 TIL 准确率 85~95% 而 CIL 只有
8~11%——精度损失几乎全部来自分类器而非特征。PTM-CL 领域的标准协议是**冻结/适配
特征 + 最近类均值（NCM）**，L2P / CODA-Prompt / InfLoRA 等均如此。改用 NCM 后：

  SimpleCIL(冻结特征) = 70.3   FOLoRA = 75.2   EWC-LoRA = 74.6

绝对数值回到文献可对照的区间，且方法的相对排序保持不变。

本脚本不重新训练，只对 `experiments/` 下每个 run 的 checkpoint 提特征 + NCM 评估。
逐 run 结果缓存在 `reports/ncm/<benchmark>/<method>__<tag>__seed<k>.json`，
已存在则跳过 —— 断电/中断后重跑同一命令即可续。

缓存的「存在即跳过」判据与重训纪律（务必一起理解）
--------------------------------------------------
缓存文件名**含 tag**，所以「读到了旧数字」只可能发生在**同一 (method, tag, seed)
被重新训练**的场合。这是有意为之：本仓库的铁律是**重训必须写新 tag，绝不覆盖旧
run 目录**（旧 checkpoint 是唯一物证）。两条合起来的效果是：

  * 正常重训（换了 tag）-> 缓存键变了 -> 强制重算，**不会静默复用旧数字**；
  * 若有人图省事用**同一个 tag** 覆盖旧 run -> 缓存命中、直接跳过，
    **评估结果会永远停留在旧实现上，且没有任何提示**。

因此：**不要用同一个 tag 重跑已经评估过的 run。** 若不得不覆盖，
先删对应的 `reports/ncm/<bench>/<method>__<tag>__seed*.json`，或换 tag 重训。
（2026-10-03 更新：上面这条纪律现在有机制兜底，但**仍然要守**。缓存记录里多了
`src_fp` —— checkpoint 的**内容**指纹（size + 头尾各 1 MiB 的 sha256，见
`eval_ncm.source_fingerprint`）。命中缓存时会比对，不符则判 stale 并重算，因此
「同 tag 原地覆盖」不再会静默沿用旧数字。这里**没有**改用 mtime：mtime 会因拷贝、
备份还原、git checkout 而变，在多进程/时钟漂移下也不可预测，作为缓存判据代价大于
收益；内容指纹是跨机器稳定的，没有这个毛病，逐 run 只多读 2 MiB。
旧缓存没有 `src_fp` 字段，按「无法判断」沿用并提示，否则会一举作废全部历史缓存、
触发整套 GPU 重评估。）

用法
----
  python -m scripts.eval_ncm_sweep --benchmark cifar100
  python -m scripts.eval_ncm_sweep --benchmark imagenetr
  python -m scripts.eval_ncm_sweep --benchmark cifar100 --methods seq,ewc,olora,folora_v2,l2p,coda
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from peft_cl.utils.io import atomic_write_json, read_json_or_none
from scripts.eval_ncm import BENCHMARK_CLASSES, evaluate_one, source_fingerprint

# 主表里出现的方法（顺序即表格顺序，与 05_experiments.tex 一致）
MAIN_METHODS = ["seq", "ewc", "olora", "l2p", "coda", "folora_v2"]
# 消融/调优配置（在 CIFAR 上需要额外一行 EWC 调优）
EXTRA_METHODS = ["folora", "folora_v2"]
# 实际扫描的方法集：**必须包含 inflora**。它的训练在 run_inflora_queue 里，
# 若这里漏了，InfLoRA 跑完也不会被评估——主表那一行会静默空着，不报错。
# 统一走这一个常量，避免「训练脚本加了方法、评估脚本没加」这类不一致。
SWEEP_METHODS = MAIN_METHODS + ["folora", "inflora"]


def discover_runs(root: Path, benchmark: str, methods):
    """扫描 experiments/<benchmark>/<method>/<tag>/seed<k>/，返回 [(method, tag, seed, dir)]。"""
    out = []
    base = root / benchmark
    if not base.exists():
        return out
    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir() or method_dir.name not in methods:
            continue
        for tag_dir in sorted(method_dir.iterdir()):
            if not tag_dir.is_dir():
                continue
            for seed_dir in sorted(tag_dir.glob("seed*")):
                if not (seed_dir / "results.json").exists():
                    continue
                try:
                    d = json.loads((seed_dir / "results.json").read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not d.get("finished"):
                    continue
                # 目录名形如 seed0；但归档/临时目录（如 seed0.failed_devbug_20260925）
                # 也会被 `glob("seed*")` 命中。**必须在这里跳过而不是让它抛异常**：
                # 本函数是被 supervise_all 的 12 个 NCM 评估阶段共同调用的，一个杂目录
                # 抛 ValueError 会让**全部**评估阶段报「pending 检查异常」并返回 0，
                # 于是新 run 永远不会被评估——正是本仓库 S-11 记录的「run 跑完但从未评估」
                # 那种静默失败，而且没有任何训练侧报错提示。2026-09-25 实际踩到过。
                if not seed_dir.name[4:].isdigit():
                    continue
                seed = int(seed_dir.name[4:])
                out.append((method_dir.name, tag_dir.name, seed, seed_dir, d))
    return out


def cache_path(cache_root: Path, benchmark, method, tag, seed):
    return cache_root / benchmark / f"{method}__{tag}__seed{seed}.json"


def main():
    p = argparse.ArgumentParser(description="NCM 全量重评估 + 主表聚合（零训练）")
    p.add_argument("--benchmark", default="cifar100", choices=["cifar100", "imagenetr"])
    p.add_argument("--exp_root", default="experiments")
    p.add_argument("--cache_root", default="reports/ncm")
    p.add_argument("--methods", default=",".join(SWEEP_METHODS))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out", default=None, help="聚合主表 json 输出路径")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    exp_root, cache_root = Path(args.exp_root), Path(args.cache_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = discover_runs(exp_root, args.benchmark, methods)
    print(f"发现 {len(runs)} 个已完成 run（{args.benchmark}，方法 {methods}）", flush=True)

    flat = []
    frozen_done = False
    for method, tag, seed, run_dir, res in runs:
        cp = cache_path(cache_root, args.benchmark, method, tag, seed)
        # 缓存读必须容错：断电可能把它写成截断的 JSON，若直接 json.loads 抛异常
        # 会连带整个队列崩掉（后面的实验全不跑）。读不出来就当未完成重算。
        rec = read_json_or_none(cp) if cp.exists() else None
        # 内容指纹（P2-5）。缓存名已含 tag，正常重训换 tag 即换缓存键；这里防的是
        # **同一 tag 原地覆盖**那条例外路径 —— 那时缓存键不变，不比对就会静默沿用
        # 旧实现算出的数字。指纹是内容派生的，不依赖时钟，故不触碰下方 docstring
        # 对 mtime 的顾虑。旧缓存没有 src_fp：按「无法判断」沿用，只提示一次，
        # 以免作废 reports/ 下已有的两百多条缓存（那会触发整套 GPU 重评估）。
        src_fp = source_fingerprint(run_dir)
        if rec is not None:
            old_fp = rec.get("src_fp")
            if src_fp is not None and old_fp is not None and old_fp != src_fp:
                print(f"[stale] {method}/{tag}/seed{seed}: checkpoint 已变"
                      f"（缓存 size={old_fp.get('size')} -> 实为 {src_fp['size']}），重算。",
                      flush=True)
                rec = None
            elif src_fp is not None and old_fp is None:
                print(f"[note ] {method}/{tag}/seed{seed}: 缓存写于 src_fp 机制之前，"
                      f"内容无法核对——本次沿用；重训请务必换 tag。", flush=True)
        if rec is not None:
            print(f"[cached] {method}/{tag}/seed{seed}: "
                  f"ACC={rec['final_acc_cil']*100:.2f} FGT={rec['forgetting_cil']*100:.2f}",
                  flush=True)
        else:
            print(f"[run   ] {method}/{tag}/seed{seed} ...", flush=True)
            try:
                out = evaluate_one(args.benchmark, res["config"]["num_tasks"], seed, device,
                                   run_dir=run_dir, data_root="data",
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers)
            except Exception as e:  # 单个 run 失败不阻断整批
                print(f"  !! 失败: {type(e).__name__}: {e}", flush=True)
                continue
            rec = {
                "method": method, "tag": tag, "seed": seed, "run_dir": str(run_dir),
                # 记录**刚刚评估过的**那份 checkpoint 的指纹（重新取一次，而不是复用
                # 循环开头那次：万一评估期间文件被换掉，要记的是实际读过的那个）。
                "src_fp": source_fingerprint(run_dir),
                "num_tasks": res["config"]["num_tasks"],
                "final_acc_til": res.get("final_acc_til"),
                "forgetting_til": res.get("forgetting_til"),
                "head_acc_cil": res.get("final_acc_cil"),
                "head_forgetting_cil": res.get("forgetting_cil"),
                **{k: v for k, v in out.items() if k != "acc_cil"},
            }
            cp.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(cp, rec)
            print(f"  ACC={rec['final_acc_cil']*100:.2f} FGT={rec['forgetting_cil']*100:.2f}",
                  flush=True)
        flat.append(rec)

        # 冻结特征 SimpleCIL 只依赖 benchmark（NCM 的类别均值与类顺序无关），只跑一次
        if not frozen_done and method == runs[0][0] and seed == runs[0][2]:
            fcp = cache_path(cache_root, args.benchmark, "simplecil", "frozen", seed)
            # 同上：截断的缓存要当未完成重算，不能只看 exists()
            frec = read_json_or_none(fcp)
            if frec is None:
                print("[run   ] simplecil (冻结特征) ...", flush=True)
                fo = evaluate_one(args.benchmark, res["config"]["num_tasks"], seed, device,
                                  run_dir=None, data_root="data",
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
                frec = {"method": "simplecil", "tag": "frozen", "seed": seed,
                        "run_dir": None, "num_tasks": res["config"]["num_tasks"],
                        **{k: v for k, v in fo.items() if k != "acc_cil"}}
                atomic_write_json(fcp, frec)
                print(f"  ACC={frec['final_acc_cil']*100:.2f} FGT={frec['forgetting_cil']*100:.2f}",
                      flush=True)
            # 2026-09-26 修：`flat.append(frec)` 原先嵌在上面那个 `if frec is None:` 分支**内部**，
            # 于是只在**首次** sweep（缓存不存在）时进表；此后每次重跑缓存都在 → 直接跳过该分支
            # → 聚合出的 summary **静默少掉 simplecil/frozen 这一行**。而它正是论文主表的
            # 「冻结特征地板行」（CIFAR 70.31 / INR 49.71），也是判「某方法是否低于地板」的基准。
            # 数字本身没丢（逐 run 缓存里都在），丢的是主表那一行，且**不报错**——同 §26.8.1 的
            # 「跑完了但从未评估」一类。改为无论数据来自缓存还是新算都 append。
            flat.append(frec)
            frozen_done = True

    # ---------------- 聚合 ----------------
    groups = {}
    for rec in flat:
        groups.setdefault((rec["method"], rec["tag"]), []).append(rec)

    agg = {}
    for (method, tag), recs in groups.items():
        accs = [r["final_acc_cil"] for r in recs]
        fgts = [r["forgetting_cil"] for r in recs]
        agg[f"{method}/{tag}"] = {
            "method": method, "tag": tag, "n": len(recs),
            "acc_mean": statistics.fmean(accs),
            "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "fgt_mean": statistics.fmean(fgts),
            "fgt_std": statistics.stdev(fgts) if len(fgts) > 1 else 0.0,
            # 2026-09-25 修：`simplecil/frozen` 组的记录**没有** `head_acc_cil` 字段
            # （frozen 记录由 `res["config"]` 拼出，不含该键），过滤后列表为空，
            # `statistics.fmean([])` 抛 `StatisticsError: fmean requires at least one
            # data point`，把**整个聚合阶段**打断 → summary 文件写不出来。
            # 崩溃点在聚合（每 run 的 json 早已 atomic_write_json 落盘），所以**数字不会丢**，
            # 只是 summary 缺；但 INR 的 5c/5 阶段因同样的 frozen 组**必然复发**，故加守卫。
            # 空列表返回 None 而不是 0.0：0.0 会被误读成「该组旧口径精度为 0」，
            # 是**编造数据**；None 明确表示「该组无此字段」。全仓库无 head_acc_mean 消费者。
            "head_acc_mean": (statistics.fmean(h) if (h := [
                r["head_acc_cil"] for r in recs if r.get("head_acc_cil") is not None]) else None),
            "seeds": sorted(r["seed"] for r in recs),
        }

    print("\n" + "=" * 78)
    print(f"{args.benchmark}  NCM 协议主表（mean ± std over seeds）")
    print("=" * 78)
    print(f"{'method/tag':<28} {'n':>3} {'ACC':>16} {'FGT':>16}  seeds")
    for key, a in sorted(agg.items(), key=lambda kv: -kv[1]["acc_mean"]):
        print(f"{key:<28} {a['n']:>3} "
              f"{a['acc_mean']*100:>7.2f}±{a['acc_std']*100:<6.2f} "
              f"{a['fgt_mean']*100:>7.2f}±{a['fgt_std']*100:<6.2f}  {a['seeds']}")

    out_path = args.out or f"reports/ncm_summary_{args.benchmark}.json"
    atomic_write_json(out_path, agg)
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
