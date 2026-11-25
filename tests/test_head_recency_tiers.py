"""护栏：§5.1 的「任务新近度三档」数字必须等于 reports/head_recency_tiers.json。

为什么值得一条独立测试
----------------------
R8-4 的回答靠的是一个**分档**数字（最后一个任务 / 倒数第二个 / 其余十八个）。分档
最容易犯的错不是算错，而是**切错边界**：第一版草稿引的 4.2% 实际是「除最后一个之外」
的平均（first_19），却被写成「前十八」。两者只差一个任务，量却差四倍，而两个数都
「看着合理」。所以这里不查文件存在，而是：

1. 从各 run 的 `results.json` 重算三档，与落盘的 JSON 逐位相等；
2. 论文 §5.1 引的四个 FOLoRA 档位数（两个基准各三档，其中最后一个任务两侧都要）
   必须原样出现在源文件里——改了数据不改文、或改了文不改数据，都会红。
"""

import json
from pathlib import Path

import pytest

from scripts.head_recency_tiers import ROWS, per_seed, tier_stats

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports" / "head_recency_tiers.json"
SECTION = REPO / "paper" / "sections" / "05_experiments.tex"

BENCHES = ("cifar100", "imagenetr")


def _recompute() -> dict:
    out = {}
    for bench in BENCHES:
        out[bench] = {}
        for label, method, t_c, t_i in ROWS:
            tag = t_c if bench == "cifar100" else t_i
            runs = per_seed(bench, method, tag)
            assert runs, f"{bench}/{method}/{tag} 没有 run"
            out[bench][label] = {"method": method, "tag": tag, **tier_stats(runs)}
    return out


def test_report_matches_runs():
    assert REPORT.exists(), "reports/head_recency_tiers.json 缺失，先跑 scripts/head_recency_tiers.py"
    stored = json.loads(REPORT.read_text(encoding="utf-8"))["benchmarks"]
    fresh = _recompute()
    for bench in BENCHES:
        for label in fresh[bench]:
            for key in ("last", "second_last", "first_18", "first_19", "all", "n_seeds"):
                assert stored[bench][label][key] == fresh[bench][label][key], \
                    f"{bench} {label} {key}: 落盘 {stored[bench][label][key]} vs 现算 {fresh[bench][label][key]}"


def test_ours_row_agrees_with_table2():
    """`all` 必须等于 Table 2 的 Head 栏（同一个量：终态学习头在全类上的准确率）。"""
    stored = json.loads(REPORT.read_text(encoding="utf-8"))["benchmarks"]
    for bench in BENCHES:
        summ = json.loads((REPO / "reports" / f"ncm_summary_{bench}.json").read_text(encoding="utf-8"))
        for label, rec in stored[bench].items():
            key = f"{rec['method']}/{rec['tag']}"
            if key in summ:
                want = summ[key]["head_acc_mean"] * 100
                assert abs(want - rec["all"]) < 0.02, f"{bench} {label}: 分档 all={rec['all']} vs Table2={want:.2f}"


@pytest.mark.parametrize("bench", BENCHES)
def test_section_quotes_the_tiers(bench):
    """FOLoRA 的三档原文必须出现在 §5.1（防止「切错边界」那类错再溜过去）。"""
    stored = json.loads(REPORT.read_text(encoding="utf-8"))["benchmarks"][bench]["FOLoRA (ours)"]
    tex = SECTION.read_text(encoding="utf-8")
    for key in ("last", "second_last", "first_18"):
        s = f"{stored[key]:.1f}\\%"
        assert s in tex, f"§5.1 里找不到 {bench} 的 {key}={s}（JSON 改了但正文没跟上？）"
