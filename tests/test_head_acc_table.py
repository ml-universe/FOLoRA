"""护栏：Table 2（`tab_head_acc.tex`）的每一格都必须等于 ncm_summary 的现行数值。

为什么值得一条独立测试
----------------------
这张表是**手写**的，不是 `make_paper_tables.py` 生成的，因此没有任何 `--check`
会碰到它。2026-11-08 的审计就抓到了三格旧数：O-LoRA 的 head 两列取自 `olora/default`
（pre-fix1），而同一行的 NCM 列与 Table 1 用的是 `olora/default_fix1`——同表内两列
指向不同一批 run，光是看表看不出来，因为两者的数值都落在合理范围内。

判据因此不是「文件存在」，而是：**按 Table 1 的 tag 逐个复算，四列全部逐位相等**。
tag 与 Table 1 共用 `MAIN_ROWS`，EWC 的 tag 走 `ewc_vs_ours_facts()` 的 best_tag，
避免这张表与主表再次分叉。
"""

import re
from pathlib import Path

import pytest

from scripts.make_paper_tables import (
    MAIN_ROWS,
    OURS,
    ewc_vs_ours_facts,
    load_summary,
)

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "paper" / "tables" / "tab_head_acc.tex"

BENCHES = ("cifar100", "imagenetr")


def _table_rows():
    """从 .tex 里读出 {行标签: (head_c, ncm_c, head_i, ncm_i)}。"""
    out = {}
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if "&" not in line or line.lstrip().startswith("%"):
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) != 5:
            continue
        cells = [parts[4].rstrip("\\ ").strip()]
        cells = parts[1:4] + cells
        if not all(re.fullmatch(r"\d+\.\d\d", c) for c in cells):
            continue
        out[parts[0]] = tuple(cells)
    return out


def _expected():
    """按 Table 1 的 tag 复算 {行标签: (head_c, ncm_c, head_i, ncm_i)}。"""
    facts = ewc_vs_ours_facts()
    summ = {b: load_summary(b) for b in BENCHES}
    exp = {}
    for label, method, t_c, t_i in MAIN_ROWS:
        if method == "simplecil":      # 表里没有 floor 行
            continue
        tags = {
            "cifar100": facts["cifar100"]["best_tag"] if method == "ewc" else t_c,
            "imagenetr": facts["imagenetr"]["best_tag"] if method == "ewc" else t_i,
        }
        vals = []
        for b in BENCHES:
            rec = summ[b][f"{method}/{tags[b]}"]
            vals.append("%.2f" % (rec["head_acc_mean"] * 100))
            vals.append("%.2f" % (rec["acc_mean"] * 100))
        exp[label] = tuple(vals)
    return exp


def test_table_is_parsed():
    rows = _table_rows()
    assert len(rows) == 7, f"应解析出 7 行方法，实际 {len(rows)}：{sorted(rows)}"
    assert "FOLoRA (ours)" in rows


@pytest.mark.parametrize("bench_idx", [0, 1])
def test_cells_match_summary(bench_idx):
    """bench_idx=0 查 CIFAR-100 的 Head/NCM 两列，1 查 ImageNet-R 的两列。"""
    rows = _table_rows()
    exp = _expected()
    b = BENCHES[bench_idx]
    off = 2 * bench_idx
    bad = []
    for label, got in rows.items():
        want = exp[label][off:off + 2]
        if tuple(got[off:off + 2]) != want:
            bad.append(f"{label}: 表里 {got[off:off + 2]}，按 ncm_summary 应为 {want}")
    assert not bad, f"{b} 的 head/NCM 与数据源不一致：\n  " + "\n  ".join(bad)


def test_ours_row_uses_main_tag():
    """FOLoRA 行必须与 Table 1 同一个 run（`OURS`），不能另取一个 tag。"""
    summ = load_summary("cifar100")
    assert f"{OURS[0]}/{OURS[1]}" in summ
    rows = _table_rows()
    assert rows["FOLoRA (ours)"][1] == "%.2f" % (summ[f"{OURS[0]}/{OURS[1]}"]["acc_mean"] * 100)
