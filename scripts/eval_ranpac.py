"""RanPAC（NeurIPS 2023, McDonnell et al.）复现 —— 回应 R8-4「缺 frozen/lightly-adapted ViT 方法」。

为什么补这一条
--------------
R8-4 原文要求补 SLCA / RanPAC / EASE / HiDe-Prompt 里**至少两个**，否则讲清为何不适用。
RanPAC 是这一族里最该补、也最便宜的一个：它**完全不训练**（冻结主干 + 随机投影 +
闭式岭回归），所以补它不会引入任何「训练预算不一致」的辩解空间——给它多少算力
都不影响结果，它就是零。

两种协议，回答两个不同的问题
----------------------------
1. `--frozen`（**RanPAC 的原生设定**）：冻结的 torchvision ViT-B/16（与 SimpleCIL
   基线同一主干，未做任何适配）。回答「只用冻结特征 + 强头，够不够」。
2. `--run_dir`（**同 checkpoint 换头**）：把我们表里某个已训练 run 的**终态** adapter
   当作冻结特征提取器，在它上面装 RanPAC 头。回答「我们比 SimpleCIL 高出来的那 7 个点，
   是不是只要换个更强的头就能复现」——这是 R8-4 背后真正的问题，零重训。

算法逐步照抄官方实现（`McDonnell-Research-Lab/RanPAC`, `models/ranpac.py`，2026-11-04 取）
------------------------------------------------------------------------------
1. 随机投影 $W_r \\in \\mathbb{R}^{d\\times M}$, 元素 i.i.d. $\\mathcal{N}(0,1)$，
   **不缩放**；$\\phi(f)=\\mathrm{ReLU}(fW_r)$（`M=0` 时退化为恒等，即 "RanPAC w/o RP"）。
2. 逐任务累积 $G \\leftarrow G + \\Phi_t^\\top\\Phi_t$, $Q \\leftarrow Q + \\Phi_t^\\top Y_t$，
   $Y_t$ 是**全局类数**（100/200）的 one-hot，未见类的行恒为 0。
3. 岭系数在**当前任务**的特征上做 $10^{-8}\\dots10^{8}$ 的网格搜索（官方 `optimise_ridge_parameter`）。
4. $W_o = (G + r I)^{-1} Q$，转置后按行当分类权重；前向 `relu(x @ W_r) @ W_o.T`，
   候选集限定为**已见类**（官方 `update_fc` 只建到已见类数，等价）。

两处必须写明的实现差异（都是「官方代码做不到、我们做得到且更严格」的那种）
----------------------------------------------------------------------
* **网格搜索的 80/20 切分要先打乱**。官方直接切 `Features[:n*0.8]`，而它的特征来自
  `DataLoader(shuffle=True)`，顺序本来就是随机的。我们这边特征是按类号顺序抽出来的，
  照抄「直接切」会变成「前 80% 的类训练、后 20% 的类验证」，得到的岭系数完全没有意义。
  所以这里用固定 seed 打乱后再切。
* **岭系数网格按官方路径直接解**（`--ridge_select direct`，默认）。本机 24 核上
  $10000\\times10000$ 求解实测 0.5 s，17 档一趟约 10 s，没有必要为省时间改算法。
  另提供 `--ridge_select dual`：用 Woodbury 恒等式
  $B(A^\\top A+rI)^{-1}A^\\top = (BA^\\top)(AA^\\top+rI)^{-1}$ 化到 $n_t\\times n_t$，
  **数学上完全等价**、数值上条件数更好，可作为两种实现互相校验的交叉检查。

用法
----
  py -3.11 -m scripts.eval_ranpac --frozen --benchmark cifar100 --seeds 0 1 2
  py -3.11 -m scripts.eval_ranpac --frozen --benchmark imagenetr --seeds 0 1 2
  py -3.11 -m scripts.eval_ranpac --run_dir experiments/cifar100/folora_v2/v2f_l3_k64/seed0
  py -3.11 -m scripts.eval_ranpac --sweep --benchmark cifar100 --out reports/ranpac_cifar100.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from peft_cl.backbone.vit import build_vit
from peft_cl.data.datasets import IMAGENET_MEAN, IMAGENET_STD, load_cifar
from peft_cl.data.split import ContinualSplit, make_class_order
from peft_cl.metrics.metrics import (average_incremental_accuracy,
                                     final_average_accuracy, forgetting)
from peft_cl.utils.seed import set_seed

BENCHMARK_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenetr": 200}

# 官方 `optimise_ridge_parameter` 的网格，逐字照抄
RIDGES = 10.0 ** np.arange(-8, 9)


# ---------------------------------------------------------------- 特征

def _deterministic_transform(image_size: int):
    """确定性变换：无随机翻转，保证同一 checkpoint 每次抽出的特征逐位相同。"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_split(benchmark, data_root, image_size, class_order):
    train_raw, test_raw = load_cifar(benchmark, data_root, image_size)
    det = _deterministic_transform(image_size)
    train_raw.transform = det
    test_raw.transform = det
    return ContinualSplit(train_raw, test_raw, class_order)


def extract_split_features(model, split, device, batch_size, num_workers=0):
    """在 `base_train` / `base_test` 上各抽一遍特征，**标签保持原始类号**。

    与 `scripts/eval_ncm.py` 共用同一个 `extract_features`（同口径），避免两份实现漂移。
    标签不在这里重映射：特征本身与类顺序无关，先存原始标签，换 class_order 时只重映射，
    不必重抽特征（冻结主干上 50k 张图一趟约 3 分钟，三个 seed 就是 9 分钟）。
    """
    from scripts.eval_ncm import extract_features

    print("  提取训练集特征 ...", flush=True)
    tr_f, tr_y = extract_features(model, split.base_train, device, batch_size,
                                  num_workers=num_workers, progress=True)
    print("  提取测试集特征 ...", flush=True)
    te_f, te_y = extract_features(model, split.base_test, device, batch_size,
                                  num_workers=num_workers, progress=True)
    return tr_f, tr_y, te_f, te_y


def verify_ncm(benchmark: str, tr_f, tr_y, te_f, te_y, class_order, num_tasks: int,
               classes_per_task: int, run_dir=None, seed: int = 0):
    """用**同一份缓存特征**复算 NCM，与 `reports/ncm/{bench}/` 里该 run 已发表的数对账。

    为什么必须对：'RanPAC 头在某个主干上得到 X' 这个结论，只有在「这份特征确实来自
    那个 checkpoint」时才成立。而**加载失败是静默的**——adapter 没灌进去，模型就退回
    冻结主干，算出来的数看着完全正常（本仓库 2026-10-03 就踩过 checkpoint 只存末个
    adapter 的坑）。对账的判据是：NCM 复算值必须与该 run 在 NCM 缓存里的 `final_acc_cil`
    一致。不一致就说明特征不是那个模型的，RanPAC 的数同样作废。
    """
    import json
    from pathlib import Path

    from scripts.eval_ncm import ncm_cil_eval

    if run_dir is None:
        method, tag = "simplecil", "frozen"
    else:
        run_dir = Path(run_dir)
        method, tag = run_dir.parent.parent.name, run_dir.parent.name

    expect = None
    for p in sorted(Path("reports/ncm", benchmark).glob("*.json")):
        r = json.load(open(p, encoding="utf-8"))
        if r.get("method") == method and r.get("tag") == tag and int(r.get("seed", -1)) == seed:
            expect = float(r["final_acc_cil"]) * 100
            break

    acc = ncm_cil_eval(tr_f, tr_y, te_f, te_y, class_order, classes_per_task, num_tasks)
    got = final_average_accuracy(acc) * 100
    if expect is None:
        print(f"  [NCM 对账] 缓存里没有 {method}/{tag}/seed{seed} 这条，无法对账"
              f"（复算值 {got:.2f}）")
        return {"expected": None, "got": got, "ok": None}
    ok = abs(got - expect) < 0.02
    # 用 ASCII 的 OK/FAIL：Windows 控制台是 GBK，✓/✗ 会直接抛 UnicodeEncodeError
    print(f"  [NCM 对账] {method}/{tag}/seed{seed}: 复算 {got:.2f} vs 缓存 {expect:.2f} "
          f"{'OK' if ok else 'FAIL —— 特征不是这个模型的，下面的数不可用'}")
    return {"expected": expect, "got": got, "ok": ok}


def remap_to_global(labels: torch.Tensor, class_order) -> torch.Tensor:
    """原始类号 → 全局连续索引（与 `ContinualSplit.class_to_global` 逐字同义）。"""
    flat = [c for task in class_order for c in task]
    lut = torch.full((max(flat) + 1,), -1, dtype=torch.long)
    for gi, c in enumerate(flat):
        lut[int(c)] = gi
    out = lut[labels.long()]
    if bool((out < 0).any()):
        raise RuntimeError("有样本的原始类号不在 class_order 里")
    return out


# ---------------------------------------------------------------- RanPAC

def _onehot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    out = torch.zeros(y.numel(), n_classes)
    out[torch.arange(y.numel()), y.long()] = 1.0
    return out


class RanPAC:
    """逐任务累积 G/Q 的闭式分类器。`M=0` 表示不做随机投影（官方 "RanPAC w/o RP"）。"""

    def __init__(self, in_dim: int, n_classes: int, M: int = 10000,
                 rp_seed: int = 1993, ridge_select: str = "direct"):
        g = torch.Generator().manual_seed(int(rp_seed))
        # 官方：torch.randn(in_features, M)，不缩放
        self.W_rand = torch.randn(in_dim, M, generator=g) if M > 0 else None
        self.dim = M if M > 0 else in_dim
        self.n_classes = n_classes
        self.G = torch.zeros(self.dim, self.dim)
        self.Q = torch.zeros(self.dim, n_classes)
        self.ridge_select = ridge_select
        self.ridge = None
        self.Wo = None

    def _phi(self, f):
        return f if self.W_rand is None else torch.relu(f @ self.W_rand)

    # -- 岭系数选择 ---------------------------------------------------
    def select_ridge(self, H: torch.Tensor, Y: torch.Tensor,
                     shuffle_seed: int = 0) -> float:
        n = int(H.shape[0] * 0.8)
        # 先打乱：官方特征是 DataLoader(shuffle=True) 抽出来的，本来就无序；
        # 我们的特征按类号排列，不打乱会变成「按类切分」。
        g = torch.Generator().manual_seed(int(shuffle_seed))
        perm = torch.randperm(H.shape[0], generator=g)
        H, Y = H[perm], Y[perm]
        A, Ya = H[:n], Y[:n]
        B, Yb = H[n:], Y[n:]

        best_r, best_loss = None, float("inf")
        if self.ridge_select == "dual":
            # (B A^T)(A A^T + rI)^{-1} Y == B (A^T A + rI)^{-1} A^T Y
            BA = B @ A.t()
            AA = A @ A.t()
            for r in RIDGES:
                Mc = AA.clone()
                Mc.diagonal().add_(float(r))
                try:
                    pred = BA @ torch.linalg.solve(Mc, Ya)
                except Exception:          # 极端条件下仍可能失败：跳过这一档，不影响其余
                    continue
                loss = float(F.mse_loss(pred, Yb))
                if loss == loss and loss < best_loss:      # 排除 NaN
                    best_loss, best_r = loss, float(r)
        else:
            # 官方路径：直接在 M×M 的 Gram 上解 17 次（实测一趟约 10 秒，够快就不走对偶）
            Gv = A.t() @ A
            Qv = A.t() @ Ya
            for r in RIDGES:
                Gc = Gv.clone()
                Gc.diagonal().add_(float(r))
                try:
                    Wo = torch.linalg.solve(Gc, Qv).t()
                except Exception:
                    continue
                loss = float(F.mse_loss(B @ Wo.t(), Yb))
                if loss == loss and loss < best_loss:
                    best_loss, best_r = loss, float(r)
            del Gv, Qv
        if best_r is None:
            raise RuntimeError("岭系数网格全部失败——这不是正常运行会出现的状态")
        return best_r

    # -- 累积一步 -----------------------------------------------------
    def update(self, feats: torch.Tensor, labels: torch.Tensor, task_id: int = 0):
        H = self._phi(feats)
        Y = _onehot(labels, self.n_classes)
        self.G += H.t() @ H
        self.Q += H.t() @ Y
        self.ridge = self.select_ridge(H, Y, shuffle_seed=1000 + task_id)
        Gc = self.G.clone()
        Gc.diagonal().add_(self.ridge)
        self.Wo = torch.linalg.solve(Gc, self.Q).t()      # (n_classes, dim)
        return self.ridge

    def predict_seen(self, feats: torch.Tensor, n_seen: int,
                     chunk: int = 2000) -> torch.Tensor:
        """对**已见类**（全局索引 0..n_seen-1）做 argmax，返回预测的全局类号。"""
        assert self.Wo is not None
        W = self.Wo[:n_seen]
        out = []
        for i in range(0, feats.shape[0], chunk):
            f = feats[i:i + chunk]
            out.append((self._phi(f) @ W.t()).argmax(dim=1))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


# ---------------------------------------------------------------- 评估矩阵

def ranpac_matrix(tr_f, tr_y, te_f, te_y, num_tasks, classes_per_task,
                  M, rp_seed, ridge_select="direct", verbose=True):
    """逐任务累积 + 每个边界评估全部已见任务，得到与主表同型的 20×20 矩阵。"""
    n_classes = num_tasks * classes_per_task
    head = RanPAC(tr_f.shape[1], n_classes, M=M, rp_seed=rp_seed,
                  ridge_select=ridge_select)
    acc = [[0.0] * num_tasks for _ in range(num_tasks)]
    for t in range(num_tasks):
        lo, hi = t * classes_per_task, (t + 1) * classes_per_task
        idx = ((tr_y >= lo) & (tr_y < hi)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise RuntimeError(f"任务 {t} 的训练样本为空（标签重映射有问题）")
        t0 = time.time()
        r = head.update(tr_f[idx], tr_y[idx], task_id=t)
        t_fit = time.time() - t0
        n_seen = (t + 1) * classes_per_task
        pred_all = head.predict_seen(te_f, n_seen)
        for j in range(t + 1):
            jlo, jhi = j * classes_per_task, (j + 1) * classes_per_task
            sel = ((te_y >= jlo) & (te_y < jhi)).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            acc[t][j] = float((pred_all[sel] == te_y[sel]).float().mean())
        if verbose:
            print(f"    task {t:2d}: ridge={r:g}  fit={t_fit:5.1f}s  "
                  f"own={acc[t][t]*100:5.2f}  "
                  f"avg_so_far={sum(acc[t][:t+1])/(t+1)*100:5.2f}", flush=True)
    return acc, head.ridge


# ---------------------------------------------------------------- 目标构建

def _load_run_model(run_dir: Path, benchmark: str, device):
    """按 `config.json` 重建方法结构并灌入 checkpoint（口径与 eval_ncm 一致）。"""
    from peft_cl.methods import build_method
    from peft_cl.utils.config import CLConfig
    from scripts.eval_ncm import load_lora_weights

    cfg = CLConfig.load(str(run_dir / "config.json"))
    model, _ = build_vit(cfg.backbone, BENCHMARK_CLASSES[benchmark] // cfg.num_tasks,
                         pretrained=True)
    method = build_method(cfg.method, model, cfg)
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
    if audit["n_unmatched"] or audit["n_shape_mismatch"]:
        print(f"  ⚠ 加载审计：{audit['n_unmatched']} 个键不存在、"
              f"{audit['n_shape_mismatch']} 个形状不符 —— 模型没按 checkpoint 完整重建")
    model.eval().to(device)
    return model, cfg, ckpt_order, hit, audit


def _features_for(args, benchmark, class_order, device):
    """返回 (tr_f, tr_y, te_f, te_y, meta)。特征带磁盘缓存（fp16），重跑免费。

    `tr_y`/`te_y` 返回时**已重映射为全局类索引**；缓存里存的是原始类号。
    """
    num_classes = BENCHMARK_CLASSES[benchmark]
    k = num_classes // args.num_tasks
    if args.frozen:
        tag = f"{benchmark}__frozen_vitb16"
        model, _ = build_vit("vit_b_16", k, pretrained=True)
        model.eval().to(device)
        meta = {"target": "frozen_vitb16"}
    else:
        run_dir = Path(args.run_dir)
        tag = f"{benchmark}__{run_dir.parent.parent.name}_{run_dir.parent.name}_{run_dir.name}"
        model, cfg, ckpt_order, hit, audit = _load_run_model(run_dir, benchmark, device)
        if ckpt_order is not None:
            class_order = ckpt_order
        meta = {"target": str(run_dir), "ckpt_hit": hit,
                "adapter_missing": audit["adapter_missing"]}

    cache = Path(args.cache_dir) / f"{tag}.pt"
    split = _load_split(benchmark, args.data_root, args.image_size, class_order)
    if cache.exists() and not args.no_cache:
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"  命中特征缓存 {cache.name}")
        tr_f, te_f = blob["tr_f"].float(), blob["te_f"].float()
        cached = True
    else:
        tr_f, _tr_y, te_f, _te_y = extract_split_features(
            model, split, device, args.batch_size, args.num_workers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # **只存特征，不存标签**。标签一律从数据集的 `.targets` 现取（见下）：
        # 存「重映射后的标签」会在换 class_order 时被迫重抽一次特征，而存「原始标签」
        # 又容易和加载端的重映射串成两次映射 —— 两种错法都不报错，只是数字会歪。
        torch.save({"tr_f": tr_f.half(), "te_f": te_f.half()}, cache)
        print(f"  写入特征缓存 {cache}")
        cached = False

    # 标签按 dataset 顺序现取（`extract_features` 用 shuffle=False，两者一一对应）
    tr_raw = torch.tensor(split.base_train.targets, dtype=torch.long)
    te_raw = torch.tensor(split.base_test.targets, dtype=torch.long)
    if tr_raw.numel() != tr_f.shape[0] or te_raw.numel() != te_f.shape[0]:
        raise RuntimeError(
            f"标签数与特征数对不上（{tr_raw.numel()}/{tr_f.shape[0]}、"
            f"{te_raw.numel()}/{te_f.shape[0]}）——缓存与当前数据集不是同一份")

    return (tr_f, remap_to_global(tr_raw, class_order),
            te_f, remap_to_global(te_raw, class_order), {**meta, "cached": cached})


# ---------------------------------------------------------------- 入口

# 与 `scripts/make_paper_tables.py` 的 MAIN_ROWS **逐条对齐**（tag 一个字都不能差）：
# O-LoRA / InfLoRA 必须走 `default_fix1`。`default` 是 P0-1 修复前的残次 checkpoint
# （只存了末个任务的 adapter，评估的模型 ≠ 训练的模型），拿它算出来的数与主表不是同一行。
# 2026-11-04 踩过：先写成 `inflora/default`，跑出 81.75，与主表的 72.95 差 2.8 分才看出来。
#
# **EWC 的 λ 是按基准分别调的**（`make_paper_tables` 用 `facts[bench]["best_tag"]` 现算）：
# CIFAR-100 最优 λ=300，ImageNet-R 最优 λ=1000。同一个 tag 套到两个基准上，INR 那一列
# 就是「欠调基线」的 checkpoint，与主表不是同一行 —— 与上面 InfLoRA 同一个坑，一并堵死。
SWEEP_BY_BENCH = {
    "cifar100": [
        ("seq", "default"),
        ("ewc", "ewc_lam300"),
        ("olora", "default_fix1"),
        ("inflora", "default_fix1"),
        ("folora_v2", "v2f_l3_k64"),
    ],
    "imagenetr": [
        ("seq", "default"),
        ("ewc", "ewc_lam1000"),
        ("olora", "default_fix1"),
        ("inflora", "default_fix1"),
        ("folora_v2", "v2f_l3_k64"),
    ],
}


def main():
    ap = argparse.ArgumentParser(description="RanPAC 复现（冻结特征 + 闭式岭回归）")
    ap.add_argument("--benchmark", default="cifar100", choices=list(BENCHMARK_CLASSES))
    ap.add_argument("--num_tasks", type=int, default=20)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="特征提取 batch（GPU 上可能同时在训模型，默认给小一点）")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--M", type=int, default=10000, help="随机投影维度；0 = 官方 'w/o RP'")
    ap.add_argument("--rp_seed", type=int, default=1993)
    ap.add_argument("--ridge_select", default="direct", choices=["direct", "dual"],
                    help="direct = 官方路径（M×M 上解 17 次）；dual = Woodbury 等价快路径")
    ap.add_argument("--cache_dir", default="reports/feat_cache")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--frozen", action="store_true", help="在冻结 ViT 上跑（RanPAC 原生设定）")
    ap.add_argument("--run_dir", default=None, help="把某个已训练 run 的终态 adapter 当特征提取器")
    ap.add_argument("--sweep", action="store_true", help="对主表五个方法各跑一遍（需 --benchmark 与 seed）")
    ap.add_argument("--seed", type=int, default=0, help="class_order 的 seed（--frozen / --sweep 用）")
    ap.add_argument("--seed_dir", type=int, default=None,
                    help="--sweep 时每个方法用 seed{seed_dir} 这个 run；不给则用 --seed")
    ap.add_argument("--max_tasks", type=int, default=None, help="调试：只跑前 N 个任务")
    ap.add_argument("--no_verify", action="store_true",
                    help="跳过 NCM 对账（只该在 NCM 缓存里确实没有该 run 时用）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = BENCHMARK_CLASSES[args.benchmark]
    k = num_classes // args.num_tasks
    set_seed(args.seed)
    class_order = make_class_order(num_classes, args.num_tasks, args.seed)
    T = args.num_tasks if args.max_tasks is None else args.max_tasks

    targets = []
    if args.frozen:
        targets.append(("frozen", None))
    if args.run_dir:
        targets.append(("run", Path(args.run_dir)))
    if args.sweep:
        sd = args.seed if args.seed_dir is None else args.seed_dir
        for method, tag in SWEEP_BY_BENCH[args.benchmark]:
            p = Path("experiments") / args.benchmark / method / tag / f"seed{sd}"
            if (p / "checkpoint.pt").exists():
                targets.append((f"{method}/{tag}/seed{sd}", p))
            else:
                print(f"[跳过] {p} 不存在")
    if not targets:
        ap.error("要么 --frozen，要么 --run_dir，要么 --sweep")

    out = {"benchmark": args.benchmark, "num_tasks": T, "M": args.M,
           "rp_seed": args.rp_seed, "class_order_seed": args.seed, "runs": {}}
    for name, run_dir in targets:
        print(f"\n=== {args.benchmark}: {name} (M={args.M}, rp_seed={args.rp_seed}) ===", flush=True)
        args.run_dir = str(run_dir) if run_dir is not None else args.run_dir
        tr_f, tr_y, te_f, te_y, meta = _features_for(args, args.benchmark, class_order, device)
        # 对账必须在算 RanPAC **之前**：若特征不是这个 checkpoint 的，后面的数一律作废
        chk = {"skipped": True}
        if not args.no_verify:
            chk = verify_ncm(args.benchmark, tr_f, tr_y, te_f, te_y, class_order, T, k,
                             run_dir=run_dir,
                             seed=int(run_dir.name.replace("seed", "")) if run_dir else args.seed)
            if chk["ok"] is False:
                raise SystemExit(
                    "NCM 对账失败：缓存特征与 NCM 缓存里的该 run 对不上，"
                    "说明 checkpoint 没有被正确加载（adapter 没灌进去，模型退回冻结主干）。")
        acc, last_ridge = ranpac_matrix(tr_f, tr_y, te_f, te_y, T, k,
                                        args.M, args.rp_seed, args.ridge_select)
        res = {
            "acc_cil": acc,
            "final_acc_cil": final_average_accuracy(acc),
            "forgetting_cil": forgetting(acc),
            "incremental_acc_cil": average_incremental_accuracy(acc),
            "last_ridge": last_ridge,
            "ncm_verify": chk,
            **meta,
        }
        out["runs"][name] = res
        print(f"  → ACC={res['final_acc_cil']*100:.2f}  GAP={res['forgetting_cil']*100:.2f}  "
              f"INC={res['incremental_acc_cil']*100:.2f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
