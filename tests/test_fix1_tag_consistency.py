"""护栏：P0-1 的 `_fix1` 开关在三处 tag 映射里必须一致。

为什么值得一条独立测试
----------------------
O-LoRA / InfLoRA 的 tag 后缀 `_fix1` 由 `scripts/significance.py::FIX1` 这一个常量
派生，但它被**三份不同的映射**消费，且这三份映射各自决定一件不同的事：

  * `significance.PAPER_MAIN_NCM`  -> 显著性家族（哪些对照进 Holm 家族）
  * `make_paper_tables.MAIN_ROWS`  -> Table 1/2/3 印哪一行
  * `eval_ncm_matrix.MAIN_TAGS`    -> Fig 2/3 画哪一行

只切一半的后果是**图和表用了不同一批 run**：两边数字都落在合理范围内，
编译照过、图照出，没有任何自动检查会报错。这与本仓库记过的「重训必须写新 tag」
纪律是同一类风险——错误表现为「悄悄用了不该用的那批数据」。

判据因此不是「脚本能跑」，而是「三份映射对同一个方法给出同一个 tag」。
"""

import os
from pathlib import Path

import pytest

from scripts import significance as S
from scripts.make_paper_tables import MAIN_ROWS
from scripts.eval_ncm_matrix import MAIN_TAGS

REPO = Path(__file__).resolve().parents[1]

# 受 P0-1 影响、tag 带后缀的两个方法（其余方法的 tag 与本次重训无关）。
AFFECTED = ("olora", "inflora")


def _main_rows_tag(method: str, bench: str):
    """MAIN_ROWS 里该方法在该基准上的 tag（列序：cifar100 第 3 列、imagenetr 第 4 列）。"""
    idx = 2 if bench == "cifar100" else 3
    hits = {row[idx] for row in MAIN_ROWS if row[1] == method}
    assert len(hits) == 1, f"MAIN_ROWS 里 {method} 有 {len(hits)} 种 tag：{hits}"
    return hits.pop()


def _paper_main_tag(method: str, bench: str):
    hits = {tag for _, m, tag in S.PAPER_MAIN_NCM[bench] if m == method}
    assert len(hits) == 1, f"PAPER_MAIN_NCM[{bench}] 里 {method} 有 {len(hits)} 种 tag：{hits}"
    return hits.pop()


@pytest.mark.parametrize("bench", ["cifar100", "imagenetr"])
@pytest.mark.parametrize("method", AFFECTED)
def test_fix1_tag_agrees_across_maps(method, bench):
    t_rows = _main_rows_tag(method, bench)
    t_paper = _paper_main_tag(method, bench)
    assert t_rows == t_paper, (
        f"{bench}/{method}：MAIN_ROWS 用 {t_rows!r} 而 PAPER_MAIN_NCM 用 {t_paper!r}；"
        f"表与显著性家族指向了不同一批 run。")
    if method in MAIN_TAGS[bench]:
        t_tags = MAIN_TAGS[bench][method]
        assert t_tags == [t_rows], (
            f"{bench}/{method}：Fig 2/3 的 MAIN_TAGS 用 {t_tags} 而 Table 1 用 {t_rows!r}；"
            f"图与表指向了不同一批 run。")


def test_affected_tags_carry_the_switch():
    """反向自检：确认 `_fix1` 这个常量真的**被**用上了。

    若哪天有人把后缀从映射里删回字面量 "default"，上面的相等性检查会在
    `FIX1="_fix1"` 时失败、在 `FIX1=""` 时静默通过 —— 后者就会让「已重训」这件事
    在数据层完全失效。所以这条钉子单独钉住：常量非空时，两个方法的 tag 必须带后缀。
    """
    if not S.FIX1:
        pytest.skip("FIX1 为空串（尚未切换），本条不适用")
    for bench in ("cifar100", "imagenetr"):
        for method in AFFECTED:
            for tag in (_main_rows_tag(method, bench), _paper_main_tag(method, bench)):
                assert tag.endswith(S.FIX1), (
                    f"{bench}/{method} 的 tag {tag!r} 没带 {S.FIX1!r} 后缀："
                    f"映射里大概又被写回了字面量。")


def test_tags_exist_on_disk():
    """tag 必须真的存在（防止「切到了一个没训过的 tag」）。

    实验目录不进公开仓库，故本地没有 `experiments/` 时跳过而不是失败。
    """
    exp = REPO / "experiments"
    if not exp.is_dir():
        pytest.skip("没有 experiments/（公开仓库不含 run 目录），跳过存在性检查")
    missing = []
    for bench in ("cifar100", "imagenetr"):
        for method in AFFECTED:
            for tag in (_main_rows_tag(method, bench),):
                d = exp / bench / method / tag
                if not (d.is_dir() and any(d.glob("seed*/results.json"))):
                    missing.append(f"{bench}/{method}/{tag}")
    assert not missing, "tag 在盘上不存在或没有任何完成的 seed：" + ", ".join(missing)
