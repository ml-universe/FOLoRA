"""`scripts/bound_probe_report.py` 的单元测试。

为什么值得测：这个汇总脚本是**判定 04_theory.tex 两条承诺去留**的唯一入口，而它要
处理的是探针在训练过程里写出的、可能被中断的半截 jsonl。它出错的代价不是崩溃，
而是**给出一个看起来正常的错误结论**（例如把所有边界都跳过后面报 KEEP）。
所以这里重点测「不该给结论时它不给结论」。
"""

import json

import pytest

from scripts.bound_probe_report import KEEP_MAX, WEAKEN_MAX, load_records, summarise, verdict


def _rec(task_id, b, c, a=0.01, delta=1.0, fisher=1.0):
    return {
        "task_id": task_id,
        "n_prev_tasks": task_id,
        "delta_norm": delta,
        "quad_term": b / 2.0,
        "fit_a_linear": a,
        "fit_b_quadratic": b,
        "fit_c_cubic": c,
        "linear_over_quad": a / b if b else float("nan"),
        "cubic_over_quad": c / b if b else float("nan"),
        "fisher_over_fitted": fisher,
        "r2_quadratic_only": 0.99,
        "dL_at_alpha1": b + c,
        "dL_curve": [0.0, 0.5, 1.0],
        "alphas": [0.0, 0.5, 1.0],
    }


def _write(dirpath, recs, raw_lines=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / "bound_terms.jsonl"
    if raw_lines is not None:
        p.write_text("\n".join(raw_lines), encoding="utf-8")
    else:
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs),
                     encoding="utf-8")
    return p


# --- verdict 的阈值边界：冻结规则本身的行为 ---------------------------------

def test_verdict_thresholds_are_the_frozen_ones():
    assert verdict(0.0)[0] == "KEEP"
    assert verdict(KEEP_MAX)[0] == "KEEP"                 # <= 0.10 含边界
    assert verdict(KEEP_MAX + 1e-9)[0] == "WEAKEN"
    assert verdict(WEAKEN_MAX)[0] == "WEAKEN"             # <= 0.30 含边界
    assert verdict(WEAKEN_MAX + 1e-9)[0] == "DELETE"


def test_verdict_without_data_never_claims_success():
    """没有可用边界时必须 NO-DATA，绝不能落到 KEEP。"""
    v, why = verdict(None)
    assert v == "NO-DATA"
    assert "KEEP" not in v
    assert why


# --- summarise：聚合与拒绝无效数据 -----------------------------------------

def test_summarise_median_and_max_are_separate():
    """中位数达标但最坏边界超标时，两个数都要在结果里，不能只留中位数。"""
    recs = [_rec(t, b=1.0, c=0.05) for t in range(1, 10)]
    recs.append(_rec(10, b=1.0, c=0.9))                   # 一个严重超标的边界
    s = summarise("fakerun", recs)
    assert s["n_boundaries_usable"] == 10
    assert s["r_cubic_median"] < KEEP_MAX                 # 中位数仍然达标
    assert s["r_cubic_max"] > WEAKEN_MAX                  # 但最坏情形被记录
    assert s["n_boundaries_over_weaken_max"] == 1


def test_summarise_skips_zero_b_denominator_instead_of_dividing():
    """b=0（无二阶项）的边界必须跳过：|c|/0 会变成 inf，混进中位数会毁掉结论。"""
    recs = [_rec(1, b=1.0, c=0.05), _rec(2, b=0.0, c=0.5), _rec(3, b=1.0, c=0.06)]
    s = summarise("fakerun", recs)
    assert s["n_boundaries_total"] == 3
    assert s["n_boundaries_usable"] == 2                  # b=0 的那条不算
    assert s["r_cubic_median"] == pytest.approx(0.055)
    assert all(r != float("inf") for r in s["per_boundary_r_cubic"])


def test_summarise_handles_all_zero_b_as_no_data():
    """所有边界 b=0 → 可用数 0 → verdict(None) → NO-DATA（不是 KEEP）。"""
    recs = [_rec(t, b=0.0, c=0.1 * t) for t in range(1, 5)]
    s = summarise("fakerun", recs)
    assert s["n_boundaries_usable"] == 0
    assert s["r_cubic_median"] is None
    assert verdict(s["r_cubic_median"])[0] == "NO-DATA"


def test_summarise_reports_median_linear_ratio_for_promise_2():
    recs = [_rec(t, b=1.0, c=0.05, a=0.3) for t in range(1, 6)]
    s = summarise("fakerun", recs)
    assert s["r_linear_median"] == pytest.approx(0.3)


def test_summarise_survives_missing_fields():
    """探针记录字段缺失（旧版本、或写失败）时不能抛，只能少算。"""
    recs = [{"task_id": 1}, {"task_id": 2, "fit_b_quadratic": None},
            {"task_id": 3, "fit_b_quadratic": "not-a-number"},
            _rec(4, b=1.0, c=0.02)]
    s = summarise("fakerun", recs)
    assert s["n_boundaries_usable"] == 1
    assert s["r_cubic_median"] == pytest.approx(0.02)


# --- load_records：坏行处理 -------------------------------------------------

def test_load_records_drops_truncated_last_line_but_keeps_the_rest(tmp_path):
    """训练被中断时最后一行常是半截 JSON。丢它、保留其余，并打印警告。"""
    good = json.dumps(_rec(1, b=1.0, c=0.05))
    _write(tmp_path / "probe_l10_k16" / "seed0", None,
           raw_lines=[good, '{"task_id": 2, "fit_b_quadr'])
    found = load_records(tmp_path)
    assert len(found) == 1
    run_dir, recs = found[0]
    assert run_dir.name == "seed0"
    assert len(recs) == 1 and recs[0]["task_id"] == 1


def test_load_records_ignores_empty_lines_and_missing_dirs(tmp_path):
    good = json.dumps(_rec(1, b=1.0, c=0.05))
    _write(tmp_path / "a" / "seed0", None, raw_lines=["", good, "", ""])
    assert load_records(tmp_path / "does_not_exist") == []
    found = load_records(tmp_path)
    assert len(found) == 1 and len(found[0][1]) == 1
