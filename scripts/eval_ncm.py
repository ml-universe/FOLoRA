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
import json
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

def load_lora_weights(model, ckpt_path):
    """把 checkpoint 里的可训练参数（LoRA + 头）灌进模型。返回是否命中任何键。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    own = dict(model.named_parameters())
    hit = 0
    for name, tensor in state.items():
        if name in own and own[name].shape == tensor.shape:
            with torch.no_grad():
                own[name].copy_(tensor)
            hit += 1
    return hit, ckpt.get("class_order"), ckpt.get("method_state")


# ---------------------------------------------------------------- 单个评估

def evaluate_one(benchmark, num_tasks, seed, device, run_dir=None, data_root="data",
                 batch_size=64, limit=None, num_workers=0, aggregate=None):
    num_classes = BENCHMARK_CLASSES[benchmark]
    classes_per_task = num_classes // num_tasks
    set_seed(seed)

    model, _ = build_vit("vit_b_16", classes_per_task, pretrained=True)

    class_order = make_class_order(num_classes, num_tasks, seed)
    method = None
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
        hit, ckpt_order, method_state = load_lora_weights(model, ckpt_path)
        if method is not None and method_state is not None:
            method.load_state_dict(method_state)
        if ckpt_order is not None:
            class_order = ckpt_order
        print(f"  载入 {hit} 个张量自 {ckpt_path}")

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
