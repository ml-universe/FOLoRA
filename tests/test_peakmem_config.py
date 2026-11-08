"""护栏：A10 峰值显存的实测记录必须是**主表那个模型**跑出来的。

为什么需要这条
--------------
`reports/peak_memory/folora_v2.json` 是正文 §5.6 与回复信 R1/R5/A10 引用的那个数。
它在 2026-11-08 之前是 `lora_rank=64` 跑出来的，而主表（Table 1/7）的适配器秩是 16：
同一个方法、同一个 k，显存却按 4 倍秩计。这类错误的形状很特别——脚本能跑完、
JSON 有值、数字看着合理，**没有任何检查会报错**，只是它量的不是论文描述的那个模型。

判据
----
不看「脚本跑没跑」，看「落盘的 config 与论文的配置是否同源」：
  * `num_tasks` 必须是 20（基线全部 20 任务，少于 20 的 k 对照不可与它们并排比较）；
  * FOLoRA 的 `lora_rank` 必须等于 `CLConfig` 的默认秩（= 主表用的秩）；
  * `folora_topk` / `folora_lambda` 必须等于主配置的 k=64 / λ=3。

`reports/peak_memory.json` 汇总里的 `folora_v2` 也一并检查：正文引用的是它。
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PM_DIR = REPO / "reports" / "peak_memory"

MAIN_TOPK = 64
MAIN_LAMBDA = 3.0
NUM_TASKS = 20


def _cfg():
    from peft_cl.utils.config import CLConfig
    return CLConfig()


def _extra_kwargs(method: str) -> dict:
    """把 `measure_peak_mem.EXTRA` 里的 `--k v` 列表还原成 kwargs。"""
    from scripts.measure_peak_mem import EXTRA

    kw = {}
    items = list(EXTRA[method])
    for i in range(0, len(items), 2):
        kw[items[i].lstrip("-")] = items[i + 1]
    return kw


def test_script_defaults_to_the_paper_rank():
    """脚本自己的默认值必须就是主表的秩，而不是「靠 --set 记得传」。"""
    rank = _cfg().lora_rank
    kw = _extra_kwargs("folora_v2")
    assert int(kw["lora_rank"]) == rank, (
        f"measure_peak_mem 的 FOLoRA 默认用 rank {kw['lora_rank']}，而主表用 {rank}；"
        f"不加 --set 直接跑就会得到一个不属于论文模型的峰值。")
    assert int(kw["folora_topk"]) == MAIN_TOPK
    assert float(kw["folora_lambda"]) == MAIN_LAMBDA


def _load(name: str):
    p = PM_DIR / name
    if not p.exists():
        pytest.skip(f"没有 {p.relative_to(REPO)}（A10 未跑）")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fname,topk", [
    ("folora_v2.json", MAIN_TOPK),
    ("folora_v2__k64r16.json", MAIN_TOPK),
    ("folora_v2__k16r16.json", 16),
])
def test_recorded_config_matches_the_main_configuration(fname, topk):
    d = _load(fname)
    cfg = d["config"]
    assert d["num_tasks"] == NUM_TASKS, (
        f"{fname} 只跑了 {d['num_tasks']} 个任务；基线全是 {NUM_TASKS} 任务，"
        f"行数不同的峰值不能并排比较。")
    assert cfg["lora_rank"] == _cfg().lora_rank, (
        f"{fname} 记录的 lora_rank={cfg['lora_rank']}，主表是 {_cfg().lora_rank}。")
    assert cfg["folora_topk"] == topk
    assert float(cfg["folora_lambda"]) == MAIN_LAMBDA


def test_summary_points_at_the_main_configuration():
    """`reports/peak_memory.json` 的 folora_v2 行是正文引用的汇总，必须同源。"""
    p = REPO / "reports" / "peak_memory.json"
    if not p.exists():
        pytest.skip("没有 reports/peak_memory.json")
    summary = json.loads(p.read_text(encoding="utf-8"))
    fv = summary.get("methods", {}).get("folora_v2", {})
    if "peak_alloc_mib" not in fv:
        pytest.skip("汇总里没有 folora_v2 的实测值")
    rec = _load("folora_v2.json")
    assert fv["peak_alloc_mib"] == rec["peak_alloc_mib"], (
        "汇总里的 FOLoRA 峰值与 reports/peak_memory/folora_v2.json 不一致："
        "汇总过期了，正文引用的是汇总。")
