"""每方法的显存占用估算（**纯解析，零 GPU**），补 R1 第 2 段缺的最后一列。

R1 要求「同时比较准确率、遗忘、总训练时间、重要性估计代价、**峰值显存**」。
前四列已在 `reports/resource_table.json` 与 `reports/timing_*.json` 里；峰值显存此前
没有系统化的数。这里按**参数量精确推算**，不测、不跑：

    optimizer_side = 16·P_trainable + 4·P_frozen + importance_bytes

  4·P_trainable   权重本身（fp32）
  4·P_trainable   梯度
  8·P_trainable   AdamW 的 exp_avg 与 exp_avg_sq（各 4 字节）
  4·P_frozen      O-LoRA / InfLoRA 那些**冻结**的历史 adapter 仍要驻留
  importance      EWC 的对角 Fisher + anchor；FOLoRA 的 ref 快照 + 累积低秩方向

**为什么这个量能替代「峰值显存」做方法间对照**：三种 adapter 方法的 backbone 完全
相同（ViT-B/16，冻结，bf16 171.6 MB），激活也随 batch 线性增长且与 adapter 形状无关；
真正把方法区分开的就是上面这一行。真正的激活峰值由 `profile_resources.py` 在干净
GPU 上实测，两者相加才是峰值——本脚本只负责**可精确复现的那部分**。

⚠ 别把 `P_frozen` 当成「不占显存」：冻结只是不产生梯度，权重还是要放在设备上。

用法
----
  py -3.11 -m scripts.memory_footprint
  py -3.11 -m scripts.memory_footprint --out reports/memory_footprint.json
"""

import argparse
import json
from pathlib import Path

MB = 1024 * 1024


def footprint(row: dict) -> dict:
    p_tr = int(row.get("trainable_at_taskT") or 0)
    p_fr = int(row.get("frozen_at_taskT") or 0)
    imp = int(row.get("importance_bytes") or 0)
    weights = 4 * p_tr
    grads = 4 * p_tr
    optim = 8 * p_tr
    frozen = 4 * p_fr
    total = weights + grads + optim + frozen + imp
    return {
        "trainable_params": p_tr,
        "frozen_params": p_fr,
        "weights_mb": round(weights / MB, 2),
        "grads_mb": round(grads / MB, 2),
        "adamw_state_mb": round(optim / MB, 2),
        "frozen_adapter_mb": round(frozen / MB, 2),
        "importance_mb": round(imp / MB, 2),
        "optimizer_side_mb": round(total / MB, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="方法间显存占用对照（零 GPU）")
    ap.add_argument("--resource", default="reports/resource_table.json")
    ap.add_argument("--out", default="reports/memory_footprint.json")
    args = ap.parse_args()

    src = json.load(open(args.resource, encoding="utf-8"))
    out = {}
    for method, row in src.items():
        f = footprint(row)
        f["label"] = row.get("label", method)
        f["importance_detail"] = row.get("importance_detail", "")
        out[method] = f

    hdr = (f"{'方法':12s} {'可训练':>10s} {'权重':>8s} {'梯度':>8s} "
           f"{'AdamW':>8s} {'冻结adapter':>12s} {'重要性状态':>11s} {'合计':>9s}")
    print(hdr)
    print("-" * 92)
    for m, f in out.items():
        print(f"{f['label']:12s} {f['trainable_params'] / 1e6:9.3f}M "
              f"{f['weights_mb']:7.1f} {f['grads_mb']:7.1f} "
              f"{f['adamw_state_mb']:7.1f} {f['frozen_adapter_mb']:11.1f} "
              f"{f['importance_mb']:10.1f} {f['optimizer_side_mb']:8.1f} MB")
    print("\n（不含冻结 backbone 的 bf16 权重 171.6 MB 与激活——两者对所有方法相同；"
          "相加才是峰值显存）")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
