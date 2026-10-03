"""零训练 NCM / 冻结特征基线评估（SimpleCIL 风格）。

动机
----
当前主表里所有方法都用「随任务增长的可训练线性头」做 CIL 分类。在冻结主干 +
无回放 + 20 任务的极端设置下，线性头会灾难性遗忘，导致绝对精度只有 10% 量级，
与文献（L2P/CODA 原文 CIFAR-100 约 83%）差一个数量级。PTM-CL 领域的标准做法是
**冻结特征 + 最近类均值（NCM）分类器**，它天然不遗忘，且几乎零成本。

本脚本做两件事，都不需要训练：
1. `--features frozen`   : 纯冻结 ViT-B/16 特征的 NCM（SimpleCIL 基线）。
2. `--run_dir <run目录>` : 加载某个已训练 run 的 LoRA 适配器权重，在**适配后的
   特征**上做 NCM（用于把 prompt/LoRA 基线放回忠实协议下比较）。

评估协议与主表一致：类增量（CIL，测试时任务 id 不可用），
逐任务累积类均值，在每个任务 j 的测试集上对「已见全部类」做最近类均值分类。

用法
----
  python -m scripts.eval_ncm --benchmark cifar100 --num_tasks 20 --seed 0
  python -m scripts.eval_ncm --benchmark cifar100 --run_dir experiments/cifar100/folora_v2/v2f_l300_k16/seed0
  python -m scripts.eval_ncm --benchmark all --sweep        # 扫全部已完成 run
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peft_cl.backbone.vit import build_vit
from peft_cl.data.datasets import load_cifar
from peft_cl.data.split import ContinualSplit, make_class_order
from peft_cl.metrics.metrics import (average_incremental_accuracy,
                                     final_average_accuracy, forgetting)
from peft_cl.utils.seed import set_seed

BENCHMARK_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenetr": 200}


# ---------------------------------------------------------------- 特征提取

@torch.no_grad()
def extract_features(model, dataset, device, batch_size=64, num_workers=0,
                     limit=None, progress=False):
    """提取 [CLS] 特征（heads 之前），返回 (N, d) 的 fp32 张量与标签张量。

    用确定性 transform（无随机翻转），保证类均值稳定、结果可复现。
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    feats, labels = [], []
    lens = 0
    for x, y in loader:
        x = x.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            z = model._process_input(x)
            n = z.shape[0]
            cls = model.class_token.expand(n, -1, -1)
            z = model.encoder(torch.cat([cls, z], dim=1))
            z = z[:, 0]
        feats.append(z.float().cpu())
        labels.append(y.clone())
        lens += y.numel()
        if progress and lens % (batch_size * 100) < batch_size:
            print(f"    ... {lens} samples", flush=True)
        if limit and lens >= limit:
            break
    return torch.cat(feats), torch.cat(labels)


# ---------------------------------------------------------------- NCM 评估

def ncm_cil_eval(train_feats, train_labels, test_feats, test_labels,
                 class_order, classes_per_task, num_tasks):
    """逐任务累积类均值的 CIL 评估，返回完整准确率矩阵与指标。

    标准 SimpleCIL 协议：任务 t 结束时，用任务 t 训练集的样本更新这 k 个新类的
    类均值（L2 归一化后的均值再归一化）；随后在**所有已见任务**的测试集上，对
    **已见全部类**做余弦最近类均值分类。
    """
    d = train_feats.shape[1]
    # 预归一化：NCM 在单位球面上做余弦相似度
    tr = F.normalize(train_feats, dim=1)
    te = F.normalize(test_feats, dim=1)

    # 全局类索引 -> 该类在 train/test 中的样本下标
    train_idx, test_idx = {}, {}
    for c in range(num_tasks * classes_per_task):
        train_idx[c] = (train_labels == c).nonzero(as_tuple=True)[0]
        test_idx[c] = (test_labels == c).nonzero(as_tuple=True)[0]

    means = {}                       # 全局类索引 -> 归一化类均值
    acc = [[0.0] * num_tasks for _ in range(num_tasks)]
    for t in range(num_tasks):
        # 1) 用任务 t 的训练数据估计这 k 个新类的类均值
        for c in range(t * classes_per_task, (t + 1) * classes_per_task):
            m = tr[train_idx[c]].mean(dim=0)
            means[c] = F.normalize(m, dim=0)
        # 2) 评估：对已见全部类做最近类均值分类
        seen = sorted(means)
        M = torch.stack([means[c] for c in seen])           # (n_seen, d)
        for j in range(t + 1):
            idx = test_idx_for_task(test_idx, j, classes_per_task)
            if idx.numel() == 0:
                continue
            sim = te[idx] @ M.t()                            # (n, n_seen)
            pred = torch.tensor(seen)[sim.argmax(dim=1)]
            acc[t][j] = (pred == test_labels[idx]).float().mean().item()
    return acc


def test_idx_for_task(test_idx, task_id, classes_per_task):
    return torch.cat([test_idx[c]
                      for c in range(task_id * classes_per_task,
                                     (task_id + 1) * classes_per_task)])


# ---------------------------------------------------------------- LoRA 权重加载

_ADAPTER_TASK_RE = re.compile(r"\.adapters\.(\d+)\.")


def audit_adapter_coverage(state, num_tasks):
    """检查 checkpoint 里逐任务 adapter 的覆盖情况。

    返回 `(present_task_ids, missing_task_ids)`。对 O-LoRA / InfLoRA 的完整 checkpoint
    应为 `{0..num_tasks-1}`；**若只有 `{num_tasks-1}`，即命中「只存了最后一个 adapter」
    的缺陷** —— 旧版 trainer 按 `requires_grad` 过滤存盘，而 `before_task` 把历史
    adapter 全冻结了，于是它们从未入库（详见 `src/peft_cl/utils/persist.py`）。
    这类 checkpoint 评估出来的分数是「只学了最后一个任务的模型」的分数。
    """
    ids = {int(m.group(1)) for k in state if (m := _ADAPTER_TASK_RE.search(k))}
    return ids, sorted(set(range(num_tasks)) - ids)


def load_lora_weights(model, ckpt_path, num_tasks=None):
    """把 checkpoint 里的模型参数灌进模型，并**报告**落不进去的键。

    返回 `(hit, ckpt_order, method_state, audit)`。`audit` 含：
      - n_state / n_hit：checkpoint 键数与成功拷入数；
      - unmatched：模型里不存在的键（说明模型结构没按 checkpoint 重建全）；
      - shape_mismatch：形状对不上的**非头**参数键（训练侧结构与被评估模型不符，是 bug）；
      - head_shape_mismatch：分类头形状不同（**预期内**，见下）；
      - adapter_tasks / adapter_missing：逐任务 adapter 的覆盖情况。

    **为什么要把这些都报出来**：原实现只 `print(f"载入 {hit} 个张量")`，命中数既不
    记录也不校验，所以「checkpoint 只含最后一个 adapter」这个缺陷在评估侧完全看不出来
    —— 数字照常产出、看着合理。现在把审计结果一路带到输出 json 里。

    **头为什么单列**（2026-10-03 修）：`evaluate_one` 用 `build_vit(bench, classes_per_task)`
    建模型（每任务 5 类 → 头是 5×768），而 checkpoint 存的是**训练末尾**的头（20 任务
    → 100×768）。两者必然不等，**每个 run 都会触发**。头不参与 NCM（特征在 heads
    之前取，见 `_extract_features`），所以这不是缺陷。但原先它混在 `shape_mismatch`
    里、并让「模型结构未完整重建」的告警**每次都响**——审计一旦恒响就等于不响，
    真正的问题会被淹掉。故拆成独立字段：记录不隐瞒，但不参与告警。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    own = dict(model.named_parameters())
    hit = 0
    unmatched, shape_mismatch, head_shape_mismatch = [], [], []
    for name, tensor in state.items():
        if name not in own:
            unmatched.append(name)
        elif own[name].shape != tensor.shape:
            (head_shape_mismatch if name.startswith("heads.")
             else shape_mismatch).append(
                (name, tuple(tensor.shape), tuple(own[name].shape)))
        else:
            with torch.no_grad():
                own[name].copy_(tensor)
            hit += 1
    adapter_tasks, adapter_missing = (audit_adapter_coverage(state, num_tasks)
                                     if num_tasks else (set(), []))
    audit = {
        "n_state": len(state), "n_hit": hit,
        "unmatched": unmatched[:20], "n_unmatched": len(unmatched),
        "shape_mismatch": shape_mismatch[:10], "n_shape_mismatch": len(shape_mismatch),
        "head_shape_mismatch": head_shape_mismatch[:4],
        "n_head_shape_mismatch": len(head_shape_mismatch),
        "adapter_tasks": sorted(adapter_tasks),
        "adapter_missing": adapter_missing,
    }
    return hit, ckpt.get("class_order"), ckpt.get("method_state"), audit


# ---------------------------------------------------------------- 单个评估

_FP_CHUNK = 1 << 20          # 取样块大小：1 MiB


def source_fingerprint(run_dir, chunk: int = _FP_CHUNK):
    """给 run 目录的 checkpoint 算一个**内容**指纹，供逐 run 评估缓存判断是否还对得上。

    为什么不用 mtime：mtime 会因拷贝、备份还原、git checkout 而变，也会在多进程与
    时钟漂移下不可预测（这条顾虑写在 eval_ncm_sweep 的模块 docstring 里，是对 mtime
    而言成立的）。内容指纹没有这个毛病 —— 文件不变则指纹不变，**跨机器/跨拷贝稳定**。

    为什么不做全文哈希：checkpoint 有 27~383 MB，一次 sweep 几十个 run 就是几十 GB
    的额外 I/O，而我们要的是「变化检测」不是密码学强度。取
    (size, sha256(头 chunk + 尾 chunk)) 足够：重新训练几乎必然改变 size，或改变尾部
    （最后写入的张量）。实测 P0-1 的修复就使 size 从 53 MB 跳到 195 MB，size 一项即可命中。

    **已知边界（2026-10-03 实测）**：这是取样指纹，不是全文哈希。若某个新 checkpoint
    与原文件**同样大小、且头尾各 1 MiB 完全相同**，只有中段不同，则指纹不变、会漏判。
    对 torch 的序列化 checkpoint 这实际不会发生（张量按序写出，重训几乎必然改变
    尾部或总长），但这条边界是真实的，故写明而不是含糊带过。

    返回 None 表示该目录没有 checkpoint（例如 SimpleCIL 的冻结特征缓存），
    调用方应视为「无法判断」而不是「已过期」。
    """
    ckpt = Path(run_dir) / "checkpoint.pt"
    try:
        size = ckpt.stat().st_size
    except OSError:
        return None
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(ckpt, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            # 若 size <= 2*chunk 会与头部重叠，重叠无害，只求覆盖到尾部
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return {"size": size, "sha256_ht": h.hexdigest()[:32]}


def evaluate_one(benchmark, num_tasks, seed, device, run_dir=None, data_root="data",
                 batch_size=64, limit=None, num_workers=0, aggregate=None):
    num_classes = BENCHMARK_CLASSES[benchmark]
    classes_per_task = num_classes // num_tasks
    set_seed(seed)

    model, _ = build_vit("vit_b_16", classes_per_task, pretrained=True)

    class_order = make_class_order(num_classes, num_tasks, seed)
    method = None
    load_audit = None      # 仅在 --run_dir 时有值；随结果落盘以便事后审计加载完整性
    if run_dir is not None:
        run_dir = Path(run_dir)
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            from peft_cl.utils.config import CLConfig
            cfg = CLConfig.load(str(cfg_path))
            num_tasks = cfg.num_tasks
            classes_per_task = num_classes // num_tasks
            set_seed(cfg.seed)
            model, _ = build_vit(cfg.backbone, classes_per_task, pretrained=True)
            # 必须先用 build_method 注入适配器/prompt 模块，checkpoint 的键才能对上
            from peft_cl.methods import build_method
            method = build_method(cfg.method, model, cfg)
            # O-LoRA 的 adapter 是逐任务动态创建的，需先补建出全部 T 个才能对上键名
            if cfg.method == "olora":
                for t in range(cfg.num_tasks):
                    method.before_task(t)
            elif cfg.method == "inflora":
                # 只为对齐 checkpoint 的键名补建 T 个分支；子空间设计由训练时完成
                for ml in method.multi_loras:
                    for _ in range(cfg.num_tasks):
                        ml.add_task()
        ckpt_path = run_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"没有 checkpoint: {ckpt_path}")
        hit, ckpt_order, method_state, load_audit = load_lora_weights(
            model, ckpt_path, num_tasks=cfg.num_tasks)
        if method is not None and method_state is not None:
            method.load_state_dict(method_state)
        if ckpt_order is not None:
            class_order = ckpt_order
        print(f"  载入 {hit} 个张量自 {ckpt_path}")
        # 头形状不同是**预期内**的（评估侧按 classes_per_task 建头、NCM 不读头），
        # 只做一行说明，不告警 —— 否则每个 run 都响，真问题会被淹掉。
        if load_audit.get("n_head_shape_mismatch"):
            print(f"  · 分类头形状不同（{load_audit['n_head_shape_mismatch']} 个键："
                  f"{load_audit['head_shape_mismatch'][0][1]} vs "
                  f"{load_audit['head_shape_mismatch'][0][2]}），评估侧按每任务类数建头，"
                  f"NCM 特征在 heads 之前取，**不影响本分数**。", flush=True)
        # 三类问题一律显式告警。都属「静默给出错数字」的情形，不能只靠 print 命中数。
        if load_audit["n_unmatched"] or load_audit["n_shape_mismatch"]:
            print(f"  ⚠ checkpoint 有 {load_audit['n_unmatched']} 个键在模型里不存在、"
                  f"{load_audit['n_shape_mismatch']} 个形状对不上（例："
                  f"{load_audit['unmatched'][:2] or [m[0] for m in load_audit['shape_mismatch'][:2]]}）"
                  f"——模型结构未按该 checkpoint 完整重建。", flush=True)
        # 只有逐任务 adapter 的方法（O-LoRA / InfLoRA）才有 «每个任务一个 adapter»
        # 的语义；seq/ewc/folora 共用一个 adapter，审计里的 missing 是正常现象，不报。
        if load_audit["adapter_missing"] and method is not None \
                and hasattr(method, "multi_loras"):
            print(f"  ⚠⚠ 该 checkpoint **缺少任务 "
                  f"{load_audit['adapter_missing']} 的 adapter**"
                  f"（只有 {load_audit['adapter_tasks']}）。这是旧版 trainer 按 "
                  f"requires_grad 过滤存盘造成的缺陷：历史 adapter 被冻结→未入库，"
                  f"评估时它们保持零初始化、对 ΔW 贡献为 0。"
                  f"**此分数对应的是「只学了最后一个任务的模型」，不可用于论文。**",
                  flush=True)

    # 评估期聚合方式覆盖（**零训练**诊断，只对逐任务多 adapter 的方法有效）。
    # 用于回答「O-LoRA 在 CIL 下低分是不是合并方式造成的」：同一个 checkpoint，
    # 只把 ΔW=ΣB_iA_i 换成 (1/T)ΣB_iA_i，看 NCM 变化。aggregate 只是 forward 里
    # 读的一个标量属性，不改任何张量形状，因此与 checkpoint 加载完全解耦。
    #
    # !! 危险：这是**诊断工具，不是出数工具** !!
    # 若训练用的是 sum、这里却用 mean 评估，就构成 train/test 不一致，得到的数字
    # **比不做这个诊断更糟**——它看着合理、且无法从结果文件里看出异常。要写进论文表的
    # 聚合对照，必须来自 `--batch olora_agg`（训练与评估都用 mean）。这个开关只用于
    # 定位机制（例如判断差距主要来自幅度还是来自方向），任何情况下都不要把它的输出
    # 当作 O-LoRA 的正式结果。
    if aggregate is not None:
        if method is None or not hasattr(method, "multi_loras"):
            raise ValueError(
                f"--aggregate 只支持逐任务多 adapter 的方法（olora/inflora），"
                f"当前 method={None if method is None else method.name!r}")
        for ml in method.multi_loras:
            ml.aggregate = aggregate
        print(f"  聚合方式覆盖为 {aggregate!r}")

    model.eval().to(device)

    # 用确定性 transform 重新加载数据集（load_cifar 默认给 train 加了随机翻转）
    from torchvision import transforms
    from peft_cl.data.datasets import IMAGENET_MEAN, IMAGENET_STD

    det = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_raw, test_raw = load_cifar(benchmark, data_root, 224)
    train_raw.transform = det
    test_raw.transform = det

    split = ContinualSplit(train_raw, test_raw, class_order)

    print("  提取训练集特征 ...", flush=True)
    tr_f, tr_y = extract_features(model, split.base_train, device, batch_size,
                                  num_workers=num_workers, limit=limit)
    print("  提取测试集特征 ...", flush=True)
    te_f, te_y = extract_features(model, split.base_test, device, batch_size,
                                  num_workers=num_workers, limit=limit)

    # ContinualSplit 的类顺序可能被 checkpoint 覆盖，重映射标签到全局索引
    tr_y = remap(tr_y, split)
    te_y = remap(te_y, split)

    acc = ncm_cil_eval(tr_f, tr_y, te_f, te_y, class_order,
                       classes_per_task, num_tasks)
    return {
        "acc_cil": acc,
        "final_acc_cil": final_average_accuracy(acc),
        "forgetting_cil": forgetting(acc),
        "incremental_acc_cil": average_incremental_accuracy(acc),
        # 权重加载审计：n_hit < n_state 或 adapter_missing 非空都意味着这个分数
        # 不是从训练出来的那个模型算的。落盘以便批量核查历史产物。
        "load_audit": load_audit,
    }


def remap(labels, split):
    gid = torch.tensor([split.class_to_global[int(c)] for c in range(len(split.class_to_global))])
    return gid[labels.long()]


# ---------------------------------------------------------------- 入口

def main():
    p = argparse.ArgumentParser(description="NCM / 冻结特征基线评估（零训练）")
    p.add_argument("--benchmark", default="cifar100",
                   choices=["cifar10", "cifar100", "imagenetr", "all"])
    p.add_argument("--num_tasks", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_dir", default=None, help="已训练 run 目录（含 checkpoint.pt）")
    p.add_argument("--data_root", default="data")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="调试用：只用前 N 个样本")
    p.add_argument("--out", default=None, help="结果写入的 json 路径")
    p.add_argument("--aggregate", default=None, choices=["sum", "mean"],
                   help="评估期覆盖逐任务 adapter 的合并方式（零训练诊断；"
                        "默认沿用 config.json 里的 olora_aggregate）")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmarks = (["cifar100", "imagenetr"] if args.benchmark == "all"
                  else [args.benchmark])

    out = {}
    for bm in benchmarks:
        print(f"=== {bm} ===", flush=True)
        res = evaluate_one(bm, args.num_tasks, args.seed, device,
                           run_dir=args.run_dir, data_root=args.data_root,
                           batch_size=args.batch_size, limit=args.limit,
                           aggregate=args.aggregate)
        out[bm] = res
        print(f"  ACC={res['final_acc_cil']*100:.2f}  FGT={res['forgetting_cil']*100:.2f}  "
              f"INC={res['incremental_acc_cil']*100:.2f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
