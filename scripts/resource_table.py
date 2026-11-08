"""资源开销表（参数量 / 推理开销 / 单任务训练耗时），零训练。

产生论文里那张「fixed budget」小表所需的全部数字：
- 可训练参数量：任务 1 时与任务 T 时（检验是否随任务数增长）；
- 总参数量：含冻结主干；
- 推理开销：prompt 方法会往序列里塞 token（增加 attention 长度），
  LoRA 系可在推理前把 ΔW 合并进 W0（**零额外推理开销**）——这是 LoRA 族
  相对 prompt 族的真实优势，值得写进表里；
- 单任务训练耗时：从已有 run.log 的 "task i done" 时间戳中位数估计。

用法：
  python -m scripts.resource_table --num_tasks 20
  python -m scripts.resource_table --num_tasks 20 --out reports/resource_table.json
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from peft_cl.backbone.vit import build_vit
from peft_cl.methods import build_method
from peft_cl.utils.config import CLConfig

METHODS = ["seq", "ewc", "olora", "inflora", "folora_v2", "l2p", "coda"]
LABELS = {
    "seq": "Seq-LoRA", "ewc": "EWC-LoRA", "olora": "O-LoRA", "inflora": "InfLoRA",
    "folora_v2": "FOLoRA", "l2p": "L2P", "coda": "CODA-Prompt",
}
# prompt 方法每个输入额外插入的 token 数 = prompt_length * prompt_topk，**从 run 的
# config 算**，不写死：旧版把 coda 写死成 25，而主表用的几何是 len=8/topk=5 = 40。
PROMPT_METHODS = ("l2p", "coda")

# 主表（Table tab:main）实际使用的 run tag。资源与耗时必须从**同一个 run** 读，
# 否则表里并列的数字来自不同预算（见 find_reference_run 的注释）。
REFERENCE_TAGS = {
    "seq": "default",
    "ewc": "default",
    "olora": "default",
    "inflora": "default",
    "folora_v2": "v2f_l3_k64",
    "l2p": "pilot20",
    "coda": "pool100_len8_ep20",
}


def extra_tokens(cfg, method: str) -> int:
    if method not in PROMPT_METHODS:
        return 0
    return int(cfg.prompt_length) * int(cfg.prompt_topk)


def _apply_trainability(method: str, meth) -> None:
    """按各方法自己的训练规则设置 requires_grad（必须与方法源码一致）。

    - O-LoRA（methods/olora.py:45-50）：历史 adapter 冻结，当前 adapter 的 A、B 都训练。
    - InfLoRA（methods/inflora.py:118-127）：历史 adapter 冻结；当前分支的降维矩阵
      lora_A 是设计出来的基底、冻结，只有升维矩阵 lora_B 可训练。
    旧版脚本把所有 adapter 一律设为可训练，于是 Table 4 把 19 个历史分支算进了
    「当前任务可训练参数」，R3-3 指出这一点，这里改正。
    """
    for ml in getattr(meth, "multi_loras", []):
        for prev in ml.adapters[:-1]:
            for p in prev.parameters():
                p.requires_grad = False
        cur = ml.adapters[-1]
        if method == "olora":
            for p in cur.parameters():
                p.requires_grad = True
        else:  # inflora
            for p in cur.lora_A.parameters():
                p.requires_grad = False
            for p in cur.lora_B.parameters():
                p.requires_grad = True


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------- 重要性模型显存
# **这张表里最容易被审稿人抓住的一格**，也是我们必须在论文里主动交代的代价：
# FOLoRA 的优势（训练参数不随任务数增长）不是免费的，它把重要性模型从 EWC 的
# O(d) 对角向量换成了 O(k·d) 的低秩子空间。不主动报，审稿人会自己算出来，
# 那时「隐藏劣势」比「承认代价」严重得多。
#
# 全部按 float32 计（与 state_dict 落盘一致），只用模型自己的参数形状推导，
# 没有任何硬编码常数。可以用已完成的 run 的 checkpoint.pt 体积交叉验证：
# FOLoRA k=16 实测 53.4 MB（含主干以外的全部状态），其中重要性部分 47.8 MB。
def lora_param_count(model) -> int:
    """只数 LoRA 的 A/B 参数（排除分类头）——FOLoRA 的 ref_params 只覆盖这些。"""
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            n += p.numel()
    return n


def importance_state(model, method: str, num_tasks: int, classes_per_task: int,
                     topk: int = 0, rank: int = 16) -> dict:
    """算出学完 num_tasks 个任务后，该方法必须常驻的重要性/参照状态大小。

    - EWC-LoRA：对角 Fisher + 上一任务参数快照，两者都只覆盖 **LoRA 参数**
      （不含分类头 —— ewc.py 的 `_adapter_param_names()` 只取 lora_A/lora_B）。
    - FOLoRA：每层 ref_params（(d,) 展平快照，仅 LoRA 参数）+ 每层 acc_grads
      ((k, d) 低秩方向)。大小为 12·k·d_layer，**与任务数无关**（每次 after_task
      都重新投影回 rank k），这正是 Highlights 第 4 条「cost 不随任务数增长」的出处。
    - 其余方法不维护重要性模型，为 0（O-LoRA / InfLoRA 的额外状态是**参数**本身，
      已计入 trainable 两列，不重复计）。
    """
    lora_n = lora_param_count(model)
    # 逐层 LoRA 参数量（12 层，每层同构；d 是 FOLoRA 低秩方向的行宽）
    n_layers = 0
    per_layer = None
    for name, p in model.named_parameters():
        if "lora_" in name:
            layer = name.split(".lora_")[0]
            if layer != per_layer:
                per_layer = layer
                n_layers += 1
    d_layer = lora_n // max(n_layers, 1)

    if method == "ewc":
        # **不含分类头**：ewc.py 的 _adapter_param_names() 只取 lora_A/lora_B，
        # fisher 与 ref_params 都只覆盖这两类。写成 lora_n + head_n 会多算 76,900
        # 个 float32（+0.3 MiB），且与实测的 1.47M 个数对不上。已核对源码。
        elems = 2 * lora_n                      # diag Fisher + anchor
        detail = f"diag Fisher + anchor over {lora_n:,} LoRA params"  # 5.6 MiB

    elif method.startswith("folora"):
        k = topk or rank
        elems = lora_n + n_layers * k * d_layer  # ref snapshots + rank-k directions
        detail = f"ref ({lora_n:,}) + {n_layers}x{k}x{d_layer:,} rank-{k} dirs"
    else:
        return {"importance_elems": 0, "importance_bytes": 0, "importance_detail": "none"}

    return {
        "importance_elems": elems,
        "importance_bytes": elems * 4,          # float32
        "importance_detail": detail,
        "n_lora_layers": n_layers,
        "d_layer": d_layer,
    }


def per_task_time_from_log(run_dir: Path):
    """从 run.log 解析每任务耗时（task i done 时间戳之差），返回中位秒数。"""
    log = run_dir / "run.log"
    if not log.exists():
        return None
    stamps = []
    pat = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) \| INFO \| === task (\d+)/")
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line)
        if m:
            h, mi, s, t = (int(x) for x in m.groups())
            stamps.append((t, h * 3600 + mi * 60 + s))
    if len(stamps) < 3:
        return None
    deltas = [stamps[i + 1][1] - stamps[i][1] for i in range(len(stamps) - 1)]
    deltas = [d for d in deltas if d > 0]
    return statistics.median(deltas) if deltas else None


def per_task_time_tag(run_dir: Path):
    """同一 tag 下**所有 seed** 的每任务中位耗时再取中位。

    单跑 seed0 不可靠：序列里第一个完成该方法的 run 往往带冷缓存（seq 的 seed0
    是 177 s，其余 9 个 seed 都在 125–136 s）。取跨 seed 中位既避开冷启动，
    也符合「报一个可复现的数」的要求。
    """
    vals = []
    for d in sorted(run_dir.parent.glob("seed*")):
        v = per_task_time_from_log(d)
        if v:
            vals.append(v)
    if not vals:
        return None, []
    return statistics.median(vals), vals


def find_reference_run(method: str, benchmark: str = "cifar100") -> Path | None:
    """找一个已完成的 run 目录用来读耗时（优先 20 任务、seed0）。"""
    base = Path("experiments") / benchmark / method
    if not base.exists():
        return None
    cands = []
    for tag_dir in sorted(base.iterdir()):
        for seed_dir in sorted(tag_dir.glob("seed*")):
            rj = seed_dir / "results.json"
            if not rj.exists():
                continue
            try:
                d = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("finished") and d["config"]["num_tasks"] == 20:
                cands.append(seed_dir)

    # **优先取主表实际用的那个 tag**。旧版按目录名排序取第一个，于是
    #   - FOLoRA 取到 pilot20（20 epoch / λ=300 / k=16），每任务耗时 348 s，
    #     而基线都是 5 epoch —— 时间开销被系统性高估约 2×；
    #   - CODA 取到 coda/default，而主表用的是 pool100_len8_ep20。
    # 两者都会让「开销」这一栏与主表的数字不在同一口径上。
    want = REFERENCE_TAGS.get(method, "")
    if want:
        hit = [c for c in cands if c.parent.name == want and c.name == "seed0"]
        if hit:
            return hit[0]
    pref = [c for c in cands if c.name == "seed0"]
    return (pref or cands or [None])[0]


def main():
    p = argparse.ArgumentParser(description="资源开销表（参数量/推理开销/训练耗时）")
    p.add_argument("--num_tasks", type=int, default=20)
    p.add_argument("--classes_per_task", type=int, default=5)
    p.add_argument("--benchmark", default="cifar100")
    p.add_argument("--out", default="reports/resource_table.json")
    args = p.parse_args()

    # 主干本体参数量：建一个不带任何方法的 ViT 数一遍（排除扩展分类头）。
    # 用「总参数 − 主干」得到「本方法创建的全部模块」，既含可训练也含冻结，
    # 这正是 R7-W10 要的三分里的第三类。
    _bare, _ = build_vit("vit_b_16", args.classes_per_task, pretrained=False)
    backbone_n = sum(p.numel() for n, p in _bare.named_parameters()
                     if not n.startswith("heads"))
    del _bare

    rows = {}
    for method in METHODS:
        # **必须读 run 自己的 config**：2026-11-02 R3 指出，用默认 config 建模型会把
        # CODA-Prompt 的 prompt 几何算成默认的 pool=20/len=5，而主表用的是
        # pool=100/len=8 —— 两者的可训练参数差 7 倍（99,845 vs 701,189）。
        ref = find_reference_run(method, args.benchmark)
        cfg_file = ref / "config.json" if ref else None
        if cfg_file is not None and cfg_file.exists():
            cfg = CLConfig.load(str(cfg_file))
        else:
            cfg = CLConfig(method=method, num_tasks=args.num_tasks, seed=0)
        # 分类头在任务 1 时为 classes_per_task 类；任务 T 时为 T*classes_per_task 类
        model, head = build_vit(cfg.backbone, args.classes_per_task, pretrained=True)
        meth = build_method(method, model, cfg)

        # O-LoRA/InfLoRA 的分支是逐任务创建的：任务 1 时应恰有 1 个分支
        if method in ("olora", "inflora"):
            for ml in meth.multi_loras:
                ml.add_task()
            _apply_trainability(method, meth)
        _, train_t1 = count_params(model)

        # 模拟学完 T 个任务后的状态（O-LoRA/InfLoRA 逐任务追加分支）
        if method in ("olora", "inflora"):
            for ml in meth.multi_loras:
                for _ in range(args.num_tasks - 1):
                    ml.add_task()
                _apply_trainability(method, meth)
        head.expand(args.num_tasks * args.classes_per_task)
        total_tT, train_tT = count_params(model)
        # 主干之外的全部参数：可训练 + 冻结但常驻 + 累计创建。
        # R7-W10 要求把「可训练 / 冻结 / 累计」三类分开，R3-3 指出旧表把历史的
        # 冻结 adapter 全计成了可训练。
        module_tT = total_tT - backbone_n
        frozen_tT = module_tT - train_tT

        secs, secs_all = per_task_time_tag(ref) if ref else (None, [])

        # 重要性模型常驻开销（FOLoRA 相对 EWC 的**代价**，必须主动报）。
        # FOLoRA 同时报 k=16 与 k=64，因为这两点的取舍正是 k 消融要画的 Pareto 曲线：
        # k 翻 4 倍，重要性显存翻 4 倍（47.8 -> 182.8 MiB），精度只多约 1 个点。
        imp_k = 64 if method.startswith("folora") else 0
        imp = importance_state(model, method, args.num_tasks, args.classes_per_task,
                               topk=imp_k, rank=cfg.lora_rank)
        if method.startswith("folora"):
            imp["importance_by_k"] = {
                k: importance_state(model, method, args.num_tasks,
                                    args.classes_per_task, topk=k, rank=cfg.lora_rank)
                for k in (16, 64, 128)
            }

        tok = extra_tokens(cfg, method)
        rows[method] = {
            "label": LABELS[method],
            "trainable_at_task1": train_t1,
            "trainable_at_taskT": train_tT,
            "frozen_at_taskT": frozen_tT,
            "cumulative_at_taskT": module_tT,
            "total_params": total_tT,
            "backbone_params": backbone_n,
            "grows_with_tasks": train_tT > train_t1 * 1.5,
            "extra_inference_tokens": tok,
            "per_task_seconds": secs,
            "per_task_seconds_by_seed": [round(v, 1) for v in secs_all],
            "epochs_per_task": int(cfg.epochs),
            "reference_run": str(ref) if ref else None,
            "config_used": str(cfg_file) if cfg_file and cfg_file.exists() else "defaults",
            **imp,
        }
        print(f"{LABELS[method]:<12} trainable@1={train_t1:>10,}  @T={train_tT:>10,}  "
              f"frozen@T={frozen_tT:>12,}  created={module_tT:>12,}  "
              f"extra_tok={tok:>2}  t/task={f'{secs:.0f}s' if secs else 'n/a'}")
        if imp["importance_bytes"]:
            mb = imp["importance_bytes"] / 1024 ** 2
            print(f"{'':<12}   重要性模型 = {imp['importance_elems']:>12,} 个 float32 "
                  f"= {mb:>7.1f} MB   [{imp['importance_detail']}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
