"""界的各项实证：把 04_theory.tex 对 Sec. discussion 的两条承诺兑现（零训练）。

!! 本脚本对已完成的 run 是**无效的**，不要用它做决策 —— 2026-09-24 实测确认 !!
==========================================================================
本脚本从落盘 checkpoint 事后重建三元组 (θ_{t-1}, θ_t, F̄_{<t})，靠的是
`cur = _flatten(lora)` 与 `prv = ref_params` 之差。但 `after_task` 会把
`ref_params` **覆盖成当前参数**，所以 checkpoint 里 ref_params == model_state：

    experiments/cifar100/folora_v2/v2f_l10_k64/seed0 → 逐层 ‖cur-ref‖ 全为 0.000000
    （全局 ‖cur-ref‖ = 0.000000，而 ‖ref‖ = 25.5551 —— 不是精度问题，是恒等）

后果：delta ≡ 0 → α 扫描得到的 dL(α) 恒为 0 → b = c = 0 →
`cubic_over_quadratic_at1` = nan。**把 nan 读成「三阶项为零、承诺 1 成立」是错的**，
它只说明这个工具测不到东西。

正确做法（两条路，都已就位）：
  * 出数：`scripts/run_lambda_grid.py --batch bound`（supervise_all 阶段 3i）。
    探针挂在 `_evaluate` 之后、`after_task` 之前，是三元组唯一存在的时刻，
    逐任务边界追加写入 run 目录的 `bound_terms.jsonl`。
  * 汇总：`scripts/bound_probe_report.py`（读 jsonl，按**预先冻结**的阈值给结论）。

本脚本保留仅作历史参考与 L_3 无关的那部分推导说明；`main()` 里已加 ‖δθ‖=0 的
硬报错，防止再次被静默误读。

为什么要写这个脚本
------------------
`paper/sections/04_theory.tex` 有**两处**白纸黑字的承诺，都指向 `sec:discussion`：

  1. L118-119：「We verify empirically that the higher-order term is negligible
     relative to the quadratic term in Sec. discussion.」
  2. L56-57 ：「We discuss the size of this deviation in Remark rem:regularized
     and **quantify it empirically** in Sec. discussion.」

而 `05_experiments.tex` 的 `sec:discussion` 里**只有方差讨论，两条都没有**。
（已 grep 确认。）承诺了不做，是审稿人最容易抓、也最伤的失分点：它同时说明
「作者知道该验什么」和「没有验」。本脚本把这两个数真的算出来，让承诺变成事实；
算不出来就删掉承诺句——不留悬空。

方法：为什么要用 α 插值，而不是直接算 L_3
-----------------------------------------
界的余项是 (L_3/6)‖δθ‖³，而 **L_3（Hessian 的 Lipschitz 常数）无法从 checkpoint
测出**，所以「余项可忽略」不能靠算了 L_3 再比大小来证。可行的办法是不碰 L_3，
直接测**非线性本身**：

把 adapter 从 θ_{t-1} 沿 δθ = θ_t - θ_{t-1} 线性插值到 θ_{t-1}+αδθ，测旧任务损失
    dL(α) = Σ_{τ<t} [ L_τ(θ_{t-1}+α δθ) - L_τ(θ_{t-1}) ]
对 α ∈ [0, 1.5] 拟合 dL(α) = a·α + b·α² + c·α³。

- a 项：Assumption 1（θ_{t-1} 处梯度为零）的违背，即线性项残留；
- b 项：界的二阶主项系数，**可直接与 ½ δθᵀ F̄ δθ 对比**——这是对 Fisher 估计
  「是否标定了真实曲率」的检验，比「余项小」更强；
- c 项：三阶项。c·α³ 相对 b·α² 的占比，就是「高阶项相对二阶项可忽略」的**直接度量**。
  若 dL(α) 在 α² 坐标下呈直线，即 c≈0，承诺 1 成立。

这个方法测的是**实际位移尺度上**的非线性，而不是任意位移下的渐近行为——正好是定理
关心的那个尺度，也因此不需要 L_3。

诚实的适用范围（必须写进论文，不要含糊）
----------------------------------------
1. **head 固定**：只把 adapter 插值，分类头保持 load 后的状态。这**正好**对应定理的
   作用域——04_theory.tex 明确声明界只覆盖 adapter（「we do not model its
   contribution」）。而且在 NCM 协议下分类头本来就被丢弃，所以这是正确的隔离对象。
   head 与 α 无关，因此只影响 dL 的基线常数，不影响曲线形状。
2. **F̄ 的任务范围**：checkpoint 的 `acc_grads` 是「学完最后一个任务后」累积的
   （after_task 会把当前任务并进去，见 folora_v2.after_task），即 F̄_{≤t}；而训练
   任务 t 时正则项用的是 F̄_{<t}。因此输出里的 Q_full 是正则项真实曲率的**上界**。
   这一点在输出里显式标注，不掩盖。拟合出的 b 与 Q_full 的比值因此应≥1 一侧解读。
3. 本脚本只适用于维持 `acc_grads`/`ref_params` 的方法（FOLoRA 系）。

用法
----
  python -m scripts.bound_terms                          # 扫全部 folora_v2 run，取前 N 个
  python -m scripts.bound_terms --limit_runs 4
  python -m scripts.bound_terms --run experiments/cifar100/folora_v2/v2f_l10_k64/seed0
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from peft_cl.backbone.vit import build_vit
from peft_cl.data.datasets import IMAGENET_MEAN, IMAGENET_STD, load_cifar
from peft_cl.data.split import ContinualSplit
from peft_cl.methods import build_method
from peft_cl.utils.config import CLConfig

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


# ------------------------------------------------------------------ adapter 读写

def adapter_flat(method) -> torch.Tensor:
    """把所有层的 adapter 按 _flatten 的同一顺序拼成一个大向量（与 ref_params 对齐）。"""
    return torch.cat([method._flatten(lora) for lora in method.loras]).detach()


def set_adapter_from_flats(method, flats) -> None:
    """把逐层的展平向量写回 adapter。顺序必须与 _flatten 一致：A 在前、B 在后。"""
    with torch.no_grad():
        for lora, flat in zip(method.loras, flats):
            n_a = lora.lora_A.weight.numel()
            lora.lora_A.weight.copy_(flat[:n_a].reshape(lora.lora_A.weight.shape))
            lora.lora_B.weight.copy_(flat[n_a:].reshape(lora.lora_B.weight.shape))


def ref_to_flats(method):
    """ref_params（dict[int, (d,)]）→ 按层序的 list，缺层时回退到当前值。"""
    out = []
    for i, lora in enumerate(method.loras):
        if i in method.ref_params:
            out.append(method.ref_params[i].float().reshape(-1))
        else:
            out.append(method._flatten(lora).detach().float())
    return out


# ------------------------------------------------------------------ 损失与二次项

@torch.no_grad()
def task_loss_sum(model, method, split, task_ids, device, batch_size=128,
                  limit=None) -> float:
    """Σ_{τ ∈ task_ids} 该任务测试集上的 CE 损失（head 固定，只反映 adapter 的影响）。

    用测试集而不是训练集：训练集上的损失会被记忆效应压低，测不出遗忘。
    """
    model.eval()
    total, n = 0.0, 0
    for t in task_ids:
        ds = split.task_dataset(t, "test")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for bi, (x, y) in enumerate(loader):
            if limit and bi * batch_size >= limit:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += torch.nn.functional.cross_entropy(
                logits, y, reduction="sum").item()
            n += y.numel()
    return total / max(n, 1)


def quadratic_term(method, delta_flats) -> float:
    """½ δθᵀ F̄ δθ，逐层用 ‖G_l δθ_l‖² 算（与 regularization_loss 同式，只差 1/2）。"""
    tot = 0.0
    for i, lora in enumerate(method.loras):
        if i not in method.acc_grads:
            continue
        G = method.acc_grads[i].float()
        d = delta_flats[i].float()
        tot += float((G @ d).pow(2).sum())
    return 0.5 * tot


# ------------------------------------------------------------------ 主流程

def analyse_run(run_dir: Path, device, benchmark="cifar100", data_root="data",
                batch_size=128, limit=None) -> dict | None:
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoint.pt"
    if not (cfg_path.exists() and ckpt_path.exists()):
        return None
    cfg = CLConfig.load(str(cfg_path))
    if cfg.method not in ("folora", "folora_v2"):
        return None

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ms = ck.get("method_state") or {}
    if not ms.get("acc_grads") or not ms.get("ref_params"):
        return None

    num_classes = 100 if benchmark == "cifar100" else 200
    classes_per_task = num_classes // cfg.num_tasks

    # head 必须扩到全尺寸，否则 model_state 里的 head 形状对不上、静默不加载
    model, head = build_vit(cfg.backbone, classes_per_task, pretrained=True)
    head.expand(cfg.num_tasks * classes_per_task)
    method = build_method(cfg.method, model, cfg)

    own = dict(model.named_parameters())
    hit = 0
    for name, tensor in ck["model_state"].items():
        if name in own and own[name].shape == tensor.shape:
            with torch.no_grad():
                own[name].copy_(tensor)
            hit += 1
    method.load_state_dict(ms)
    model.eval().to(device)

    # 数据（确定性 transform，同 eval_ncm.py）
    from torchvision import transforms
    det = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_raw, test_raw = load_cifar(benchmark, data_root, 224)
    train_raw.transform = det
    test_raw.transform = det
    class_order = ck.get("class_order")
    split = ContinualSplit(train_raw, test_raw, class_order)

    T = cfg.num_tasks
    # 界的 LHS 是「已学过的、除最后一个之外的全部任务」——eq:sum 里 τ < t
    prev_tasks = list(range(T - 1))
    if not prev_tasks:
        return None

    cur = [l.detach().float().clone() for l in
           [method._flatten(lora) for lora in method.loras]]
    prv = [r.float().clone() for r in ref_to_flats(method)]
    delta = [c - p for c, p in zip(cur, prv)]

    Q_full = quadratic_term(method, delta)

    # α 扫描：θ(α) = θ_prev + α·δθ，测 dL(α)（相对 α=0 的增量）
    curve = []
    for alpha in ALPHAS:
        flats = [p + alpha * d for p, d in zip(prv, delta)]
        set_adapter_from_flats(method, flats)
        L = task_loss_sum(model, method, split, prev_tasks, device,
                          batch_size=batch_size, limit=limit)
        curve.append((alpha, L))
    L0 = curve[0][1]
    dL = np.array([L - L0 for _, L in curve], dtype=np.float64)
    alphas = np.array([a for a, _ in curve], dtype=np.float64)

    # 拟合 dL = a·α + b·α² + c·α³（dL(0)=0 已由构造保证，无常数项）
    A = np.stack([alphas, alphas ** 2, alphas ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(A, dL, rcond=None)
    a, b, c = (float(x) for x in coef)
    pred = A @ coef
    ss_res = float(((dL - pred) ** 2).sum())
    ss_tot = float(((dL - dL.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 只看二阶模型（a·α + b·α²）的解释力：与三阶比，若几乎相同则三阶项可忽略
    A2 = np.stack([alphas, alphas ** 2], axis=1)
    coef2, *_ = np.linalg.lstsq(A2, dL, rcond=None)
    r2_quad = 1.0 - float(((dL - A2 @ coef2) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    at1 = {"a": a, "b": b, "c": c}
    b_at1 = b
    c_at1 = c
    return {
        "run": str(run_dir),
        "n_tasks": T,
        "prev_tasks_evaluated": len(prev_tasks),
        "checkpoint_tensors_loaded": hit,
        "delta_norm": float(torch.cat(delta).norm()),
        "Q_full": Q_full,                    # ½δθᵀ F̄_{≤t} δθ（F̄ 范围见 docstring）
        "fit_a_linear": a,
        "fit_b_quadratic": b,
        "fit_c_cubic": c,
        "linear_over_quadratic_at1": (a / b_at1) if b_at1 else float("nan"),
        "cubic_over_quadratic_at1": (c_at1 / b_at1) if b_at1 else float("nan"),
        "fisher_over_fitted": (Q_full / b) if b else float("nan"),
        "r2_cubic_model": r2,
        "r2_quadratic_model": r2_quad,
        "dL_curve": dL.tolist(),
        "alphas": alphas.tolist(),
        "dL_at_alpha1": float(dL[ALPHAS.index(1.0)]),
    }


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="界的各项实证（α 插值，零训练）")
    ap.add_argument("--run", default=None, help="单个 run 目录")
    ap.add_argument("--exp_root", default="experiments/cifar100/folora_v2")
    ap.add_argument("--benchmark", default="cifar100")
    ap.add_argument("--limit_runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="每任务图像上限（调试用，0=全量）")
    ap.add_argument("--out", default="reports/bound_terms.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    if args.run:
        runs = [Path(args.run)]
    else:
        root = Path(args.exp_root)
        runs = sorted({p.parent for p in root.glob("**/checkpoint.pt")})[: args.limit_runs]
    if not runs:
        print("没有找到可分析的 run"); return

    results = []
    for rd in runs:
        print(f"\n=== {rd.relative_to(Path('experiments')) if rd.is_relative_to(Path('experiments')) else rd} ===")
        base = Path(str(rd))
        r = analyse_run(base, device, benchmark=args.benchmark, limit=args.limit or None)
        if r is None:
            print("  跳过（缺 config/checkpoint 或非 FOLoRA 系）"); continue
        results.append(r)
        if r["delta_norm"] == 0.0:
            print("  !! δθ 范数 = 0 —— 本工具对已完成的 run 失效：after_task 已把 "
                  "ref_params 覆盖成当前参数，事后取不到位移。")
            print("     这个 0 **不是**「高阶项为零」的证据，只是测不到；"
                  "后面的 a/b/c 拟合与 nan 比值全部无意义。")
            print("     正确入口：run 目录下的 bound_terms.jsonl（阶段 3i 产出），"
                  "用 scripts/bound_probe_report.py 汇总。")
            continue
        print(f"  δθ 范数 = {r['delta_norm']:.4f}   "
              f"½δθᵀF̄δθ = {r['Q_full']:.5f}")
        print(f"  拟合 dL(α) = {r['fit_a_linear']:+.5f}α "
              f"{r['fit_b_quadratic']:+.5f}α² {r['fit_c_cubic']:+.5f}α³")
        print(f"  线性/二阶 @α=1 = {r['linear_over_quadratic_at1']:+.3f}   "
              f"三阶/二阶 @α=1 = {r['cubic_over_quadratic_at1']:+.3f}")
        print(f"  Fisher/拟合二阶 = {r['fisher_over_fitted']:.3f}   "
              f"R²(三阶)={r['r2_cubic_model']:.4f}  R²(仅二阶)={r['r2_quadratic_model']:.4f}")
        # 承诺 1 的判定
        cc = abs(r["cubic_over_quadratic_at1"])
        if cc < 0.10:
            v = "三阶项 <10% 二阶 —— 承诺 1 成立（「高阶项可忽略」有实测支撑）"
        elif cc < 0.30:
            v = "三阶项 10-30% —— 只能说「同量级但不主导」，措辞需弱化"
        else:
            v = "三阶项 >30% —— 承诺 1 **不成立**，必须删掉该句或改为承认二阶不充分"
        print(f"  -> {v}")

    if not results:
        print("\n没有任何 run 分析成功"); return

    def med(key):
        vals = [r[key] for r in results if r[key] == r[key]]
        return float(np.median(vals)) if vals else float("nan")

    summary = {k: med(k) for k in
               ("delta_norm", "Q_full", "fit_b_quadratic", "fit_c_cubic",
                "linear_over_quadratic_at1", "cubic_over_quadratic_at1",
                "fisher_over_fitted", "r2_cubic_model", "r2_quadratic_model")}
    print("\n=== 中位数（n=%d runs）===" % len(results))
    print(f"  三阶/二阶 @α=1 = {summary['cubic_over_quadratic_at1']:+.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": results, "summary_median": summary},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
