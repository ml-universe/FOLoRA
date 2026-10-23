"""Fisher 谱分析：把 FOLoRA 的「等权 vs σ² 加权」之争变成**可预注册的预测**。

为什么需要这个脚本
------------------
论文的 Highlights 第 2 条声称「importance-weighted 优于 equal-weight」。这条声明
只有在 σ_j² 本身**足够不均匀**时才可能成立：如果所有 σ_j² 几乎相等，那么加权与
等权在数学上就是同一个正则项，消融必然测不出差异。

所以在本项目的等权消融网格（run_lambda_grid 的 `unweighted` 批）跑完**之前**，
我们可以先从已完成的 run 的 `method_state.acc_grads` 直接算出 σ 的分布，
给出一个**先验预测**：该消融会不会出现可测差异。跑完再对照——预测对了，
说明「用谱形状判断加权是否重要」这套说法有预测力，可以写进论文；
预测错了，说明这套说法不能写，但至少我们提前知道，而不是把宝押在上面。

数学背景（与 folora_v2.py 的实现一一对应）
------------------------------------------
`_topk_directions` 返回 `eigvecs[:, -kk:].t() @ G`。若 G = U S Vᵀ，则
G Gᵀ = U S² Uᵀ，故 eigvecs = U，于是该返回值的第 j 行 = σ_j v_jᵀ，
即 **行范数恰好是第 j 个奇异值 σ_j**。正则项
    L = Σ_j (σ_j v_jᵀ dθ)²
把 σ_j² 作为方向 v_j 上的权重，这正是 `_equalize_row_weights` 要抹平的东西。

用法：
  python -m scripts.fisher_spectrum                       # 扫全部已完成的 folora run
  python -m scripts.fisher_spectrum --out reports/fisher_spectrum.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch


def spectrum_from_checkpoint(ckpt_path: Path) -> dict | None:
    """读一个 checkpoint 的 acc_grads，返回逐层 σ 统计。"""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    ms = ck.get("method_state") or {}
    ag = ms.get("acc_grads")
    if not ag:
        return None

    per_layer = []
    for key in sorted(ag.keys(), key=lambda s: int(s)):
        G = ag[key]
        if G is None or G.numel() == 0:
            continue
        # 行范数 = 奇异值 σ_j（见模块 docstring 的推导）
        sigma = G.float().norm(dim=1)
        sigma = sigma[sigma > 0]
        if sigma.numel() < 2:
            continue
        k_eff = int(sigma.numel())
        s2 = sigma.pow(2)
        # 两个 participation ratio：
        #   PR_sigma 在 σ 上（"方向数量"的直觉）
        #   PR_lambda 在 σ² 上（**正则项真正看到的权重分布**，才是相关的那一个）
        pr_sigma = float(sigma.sum() ** 2 / sigma.pow(2).sum())
        pr_lambda = float(s2.sum() ** 2 / s2.pow(2).sum())
        per_layer.append({
            "k_eff": k_eff,
            "sigma_max": float(sigma.max()),
            "sigma_min": float(sigma.min()),
            "sigma_ratio": float(sigma.max() / sigma.min()),
            "sigma2_ratio": float((sigma.max() / sigma.min()) ** 2),
            "pr_sigma": pr_sigma,
            "pr_sigma_frac": pr_sigma / k_eff,
            "pr_lambda": pr_lambda,
            "pr_lambda_frac": pr_lambda / k_eff,
        })
    if not per_layer:
        return None

    def med(field):
        return statistics.median(l[field] for l in per_layer)

    return {
        "k": per_layer[0]["k_eff"],
        "n_layers": len(per_layer),
        "sigma_ratio_median": med("sigma_ratio"),
        "sigma2_ratio_median": med("sigma2_ratio"),
        "pr_sigma_median": med("pr_sigma"),
        "pr_sigma_frac_median": med("pr_sigma_frac"),
        "pr_lambda_median": med("pr_lambda"),
        "pr_lambda_frac_median": med("pr_lambda_frac"),
        "per_layer": per_layer,
    }


def find_runs(exp_root: Path, method_prefix: str = "folora") -> list[Path]:
    """所有带 checkpoint 的已完成 run。"""
    out = []
    for method_dir in sorted(exp_root.iterdir()):
        if not method_dir.is_dir() or not method_dir.name.startswith(method_prefix):
            continue
        for tag_dir in sorted(method_dir.iterdir()):
            for seed_dir in sorted(tag_dir.glob("seed*")):
                cp = seed_dir / "checkpoint.pt"
                if cp.exists():
                    out.append(cp)
    return out


def main():
    # Windows 控制台默认 GBK，输出里的 σ/σ² 会直接抛 UnicodeEncodeError 把脚本打死。
    # 重配成 utf-8 + replace，宁可显示成 ? 也不要崩。
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="FOLoRA Fisher 谱分析（零训练）")
    ap.add_argument("--exp_root", default="experiments/cifar100")
    ap.add_argument("--method_prefix", default="folora")
    ap.add_argument("--out", default="reports/fisher_spectrum.json")
    args = ap.parse_args()

    root = Path(args.exp_root)
    runs = find_runs(root, args.method_prefix)
    print(f"扫描到 {len(runs)} 个带 checkpoint 的 {args.method_prefix}* run\n")

    # 按 (tag, k) 归组，方便看 k=16 与 k=64 的差别
    by_tag: dict[str, list[dict]] = {}
    for cp in runs:
        spec = spectrum_from_checkpoint(cp)
        if spec is None:
            continue
        tag = cp.parent.parent.name
        spec["run"] = str(cp.parent.relative_to(root))
        by_tag.setdefault(tag, []).append(spec)

    print(f"{'tag':<20} {'k':>4} {'n':>3} {'σmax/σmin':>10} {'σ²比':>8} "
          f"{'PR_σ/k':>8} {'PR_λ/k':>8}")
    print("-" * 70)
    summary = {}
    for tag, specs in sorted(by_tag.items()):
        for k in sorted({s["k"] for s in specs}):
            sel = [s for s in specs if s["k"] == k]
            sr = statistics.median(s["sigma_ratio_median"] for s in sel)
            s2r = statistics.median(s["sigma2_ratio_median"] for s in sel)
            psf = statistics.median(s["pr_sigma_frac_median"] for s in sel)
            plf = statistics.median(s["pr_lambda_frac_median"] for s in sel)
            print(f"{tag:<20} {k:>4} {len(sel):>3} {sr:>10.2f} {s2r:>8.1f} "
                  f"{psf:>8.3f} {plf:>8.3f}")
            summary.setdefault(str(k), []).append({
                "tag": tag, "n": len(sel), "sigma_ratio": sr, "sigma2_ratio": s2r,
                "pr_sigma_frac": psf, "pr_lambda_frac": plf,
            })

    # ---- 先验预测：加权与等权会不会有可测差异 ----
    # **重要 confound**：σ_max/σ_min 随 k 单调增大是**构造性的**（多取方向必然包含更小的
    # σ_j），所以直接比「k=64 比 k=16 更不均匀」是同义反复，审稿人会指出。
    # 有信息量的是**绝对有效方向数** n_eff = PR_λ（在 σ² 上算）：
    # 它回答「σ² 的质量实际落在几个方向上」。若 n_eff 在两个 k 下都 ≈ 10-15，
    # 说明 k 取 16 已经覆盖了绝大部分权重，多出来的方向权重极低——这既解释了
    # k 从 16 提到 64 只换来约 1 个点，也预测了等权消融会出现差异
    # （等权会给这些低权重方向不合理的放大）。
    print("\n先验预测（σ² 越不均匀，等权消融越可能测出差异）：")
    print("  注意：跨 k 比 σ² 比是同义反复（k 越大必然包含越小 σ_j）。"
          "看 n_eff = PR_λ 的绝对值。")
    for k, rows in sorted(summary.items()):
        s2 = statistics.median(r["sigma2_ratio"] for r in rows)
        plf = statistics.median(r["pr_lambda_frac"] for r in rows)
        n_eff = plf * int(k)
        if s2 < 3:
            verdict = "几乎均匀 —— 预测消融**测不出**差异（该 highlight 无支撑）"
        elif s2 < 10:
            verdict = "轻度不均匀 —— 预测差异很小（±1.3 点噪声带内，难判定）"
        else:
            verdict = "显著不均匀 —— 预测消融**能测出**差异（bonus: 谱可预测）"
        print(f"  k={k:>3}: σ²比中位数 {s2:>6.1f}, PR_λ/k = {plf:.3f}, "
              f"n_eff ≈ {n_eff:>4.1f}  ->  {verdict}")
    print("\n  若两个 k 的 n_eff 接近（≈10-15），说明 k=16 已覆盖主要 σ² 权重，"
          "这是「紧凑子空间足够」的定量依据，也解释了 k 加倍只小幅涨点。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"by_tag": {t: s for t, s in by_tag.items()}, "summary": summary},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
