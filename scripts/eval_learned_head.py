"""用**训练时联合学习的分类头**评估已完成的 run（零训练，回应 R8-3 / R8-10）。

为什么需要它
------------
主表所有方法都用「冻结特征 + 最近类均值（NCM）」分类。R8-3 质疑的是这个口径本身：
「L2P / CODA-Prompt / InfLoRA 在原文里用的是各自学习的分类器，不是 adapted features
上的 NCM 头。请核实这句话，并在**标准学习头协议**下至少报告主要基线。」

这个质疑不能靠写一段说明消掉——但也不需要重训：**每个 checkpoint 里都存着训练时
联合学习的分类头**（`heads.head.weight/bias`，形状 = 已见类数 × 768）。`eval_ncm.py`
为了复用 NCM 特征一直按「每任务类数」建头，于是每个 run 都报一次头形状不匹配、把这
个头丢掉了。本脚本反过来：**按总类数建头**，于是头正好对上、原样载入，再用它做
CIL / TIL 分类。

与 NCM 的差别因此只有一个：分类器是训练中学出来的线性头（会遗忘），而不是每任务
重算的类均值（不遗忘）。两者的差就是 R8-3 要的那个数。

用法
----
  py -3.11 -m scripts.eval_learned_head --run experiments/cifar100/seq/default/seed0
  py -3.11 -m scripts.eval_learned_head --preset main --out reports/learned_head_cifar100.json

注意
----
* 只支持 adapter 类方法（seq / ewc / folora / olora / inflora）；prompt 类（l2p / coda）
  没有 `heads.head`，传进来会直接报错而不是给一个看着合理的错数。
* 必须按 **num_classes** 建头。若沿用 eval_ncm 的 classes_per_task，头会载不进去，
  而 `load_lora_weights` 只会把它记成 head_shape_mismatch 然后**静默跳过**——
  那就变成用随机初始化的头评估，数字会接近 1/100。
"""

import argparse
import json
from pathlib import Path

import torch

from peft_cl.backbone.vit import build_vit
from peft_cl.data.datasets import load_cifar
from peft_cl.data.split import ContinualSplit, make_class_order
from peft_cl.methods import build_method
from peft_cl.utils.config import CLConfig
from peft_cl.utils.seed import set_seed
from scripts.eval_ncm import (BENCHMARK_CLASSES, extract_features,
                              load_lora_weights, remap)

# 主表各行（方法, tag），与论文 Table 1 的 CIFAR-100 列一一对应
PRESET_MAIN = [
    ("seq", "default"),
    ("ewc", "ewc_lam300"),
    ("olora", "default_fix1"),
    ("inflora", "default"),
    ("folora_v2", "v2f_l3_k64"),
]


def eval_one(run_dir: Path, device: torch.device, data_root: str = "data",
             batch_size: int = 64, num_workers: int = 0, limit=None) -> dict:
    cfg = CLConfig.load(str(run_dir / "config.json"))
    num_classes = BENCHMARK_CLASSES[cfg.benchmark]
    set_seed(cfg.seed)

    # **关键**：头按总类数建，checkpoint 里那个 100×768 的头才能原样载入。
    model, _ = build_vit(cfg.backbone, num_classes, pretrained=True)
    method = build_method(cfg.method, model, cfg)
    # O-LoRA / InfLoRA 的 adapter 是逐任务动态建的，必须先补建 T 个才能对上键名
    if cfg.method == "olora":
        for t in range(cfg.num_tasks):
            method.before_task(t)
    elif cfg.method == "inflora":
        for ml in method.multi_loras:
            for _ in range(cfg.num_tasks):
                ml.add_task()

    hit, ckpt_order, method_state, audit = load_lora_weights(
        model, run_dir / "checkpoint.pt", num_tasks=cfg.num_tasks)
    if method_state is not None:
        method.load_state_dict(method_state)

    # 头必须真的载进来。载不进来就是「用随机头评估」，宁可失败也不出数。
    if audit.get("n_head_shape_mismatch"):
        raise RuntimeError(
            f"{run_dir}: 分类头形状仍不匹配 {audit['head_shape_mismatch'][:2]}——"
            f"按 num_classes={num_classes} 建头本应完全对上，说明该 checkpoint 的头"
            f"不是按已见类数增长的，不能用本脚本评估")
    if audit.get("n_unmatched") or audit.get("n_shape_mismatch"):
        raise RuntimeError(f"{run_dir}: checkpoint 与模型结构不符 {audit}")

    # 必须 .to(device)：`extract_features` 里 `x = x.to(device)`，模型若留在 CPU，
    # 第一层 conv 就拿 bf16 输入去撞 fp32 权重/bias（报的就是那个错），看起来像 dtype
    # 问题、实则是设备没搬。eval_ncm.evaluate_one 有这一行，这里原先漏了。
    model.eval().to(device)

    head = getattr(getattr(model, "heads", None), "head", None)
    if head is None:
        raise RuntimeError(f"{run_dir}: method={cfg.method} 没有 heads.head，"
                           f"本脚本只支持 adapter 类方法")

    class_order = ckpt_order or make_class_order(num_classes, cfg.num_tasks, cfg.seed)
    from torchvision import transforms
    from peft_cl.data.datasets import IMAGENET_MEAN, IMAGENET_STD
    det = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_raw, test_raw = load_cifar(cfg.benchmark, data_root, 224)
    train_raw.transform = det
    test_raw.transform = det
    split = ContinualSplit(train_raw, test_raw, class_order)
    te_f, te_y = extract_features(model, split.base_test, device, batch_size,
                                  num_workers=num_workers, limit=limit)
    te_y = remap(te_y, split)

    # extract_features 在 autocast 下产物可能是 bf16；头是 fp32，必须显式升精度，
    # 否则报 "Input type (BFloat16) and bias type (float) should be the same"。
    with torch.no_grad():
        logits = head(te_f.to(device).float()).float().cpu()
    pred = logits.argmax(dim=1)
    acc_cil = float(pred.eq(te_y).float().mean())

    cpt = num_classes // cfg.num_tasks
    til = []
    for j in range(cfg.num_tasks):
        lo, hi = j * cpt, (j + 1) * cpt
        idx = (te_y >= lo) & (te_y < hi)
        if not bool(idx.any()):
            til.append(0.0)
            continue
        p = logits[idx, lo:hi].argmax(dim=1) + lo
        til.append(float(p.eq(te_y[idx]).float().mean()))
    acc_til = sum(til) / len(til)

    # 与 trainer 自己记的那一列并排，用来交叉验证本脚本的载入路径没走偏。
    # **注意名字**：`results.json` 的 `final_acc_cil` **本来就是学习头结果**
    # （metrics.evaluate 在 class_range=None 时对 100 维 logits 取全头 argmax，
    # trainer._evaluate 每个任务边界都这么算），不是 NCM。2026-11-04 这里一度写成
    # `ncm_final_acc_cil_growing_head`，会让人以为主表 NCM 口径下还有另一个数。
    trainer_cil = None
    res_path = run_dir / "results.json"
    if res_path.exists():
        trainer_cil = json.load(open(res_path, encoding="utf-8")).get("final_acc_cil")

    return {"run": str(run_dir), "method": cfg.method, "tag": cfg.tag,
            "seed": cfg.seed, "learned_acc_cil": acc_cil,
            "learned_acc_til": acc_til, "per_task_cil": til,
            "trainer_final_acc_cil": trainer_cil,
            "audit": {k: audit[k] for k in
                      ("n_state", "n_hit", "n_unmatched", "n_shape_mismatch",
                       "n_head_shape_mismatch")}}


def main() -> None:
    p = argparse.ArgumentParser(description="学习头协议评估（零训练）")
    p.add_argument("--run", action="append", default=[],
                   help="run 目录（可重复）。形如 experiments/cifar100/folora_v2/v2f_l3_k64/seed0")
    p.add_argument("--preset", choices=["main"], default=None,
                   help="main = 主表五个方法的 CIFAR-100 行（seeds 0-2）")
    p.add_argument("--benchmark", default="cifar100")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data_root", default="data")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="reports/learned_head_eval.json")
    args = p.parse_args()

    runs = [Path(r) for r in args.run]
    if args.preset == "main":
        for method, tag in PRESET_MAIN:
            for s in args.seeds:
                runs.append(Path("experiments") / args.benchmark / method / tag / f"seed{s}")
    if not runs:
        p.error("要么给 --run，要么给 --preset main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, failed = [], []
    for r in runs:
        if not (r / "checkpoint.pt").exists():
            print(f"跳过（无 checkpoint）: {r}", flush=True)
            failed.append({"run": str(r), "reason": "no checkpoint"})
            continue
        try:
            row = eval_one(r, device, args.data_root, args.batch_size,
                           args.num_workers, args.limit)
            rows.append(row)
            print(f"{row['method']}/{row['tag']} seed{row['seed']}: "
                  f"learned_head CIL={row['learned_acc_cil']:.2f} "
                  f"TIL={row['learned_acc_til']:.2f} "
                  f"(trainer 记的 CIL={row['trainer_final_acc_cil']})", flush=True)
        except Exception as e:
            print(f"失败 {r}: {type(e).__name__}: {e}", flush=True)
            failed.append({"run": str(r), "reason": f"{type(e).__name__}: {e}"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "failed": failed}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n写入 {out}（成功 {len(rows)}，失败 {len(failed)}）")


if __name__ == "__main__":
    main()
