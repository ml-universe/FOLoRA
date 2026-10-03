"""显著性检验脚本单元测试（纯 CPU，不读 experiments/）。"""

import json

from scripts.significance import compare, find_key, run_test, values


def _seeds(accs, fgts):
    """构造 {seed: results} 的最小样本。"""
    return {i: {"final_acc_cil": a, "forgetting_cil": f}
            for i, (a, f) in enumerate(zip(accs, fgts))}


def test_values_sorted_by_seed():
    seeds = {2: {"final_acc_cil": 0.2}, 0: {"final_acc_cil": 0.1}, 1: {"final_acc_cil": 0.3}}
    assert values(seeds, "final_acc_cil") == [10.0, 30.0, 20.0]


def test_identical_groups_not_significant():
    a = [10.0, 11.0, 9.0, 10.5]
    _, p, _, _, _ = run_test(a, list(a), paired=False)
    assert p > 0.99


def test_clear_effect_is_significant():
    a = [20.0, 21.0, 19.5, 20.5]
    b = [10.0, 11.0, 9.5, 10.5]
    _, p, _, _, _ = run_test(a, b, paired=False)
    assert p < 0.001


def test_paired_and_welch_can_differ():
    # 两组同向但配对后更一致 → 配对检验的 p 更小
    a = [10.0, 20.0, 30.0, 40.0]
    b = [9.0, 19.0, 29.0, 39.0]
    _, p_welch, _, _, _ = run_test(a, b, paired=False)
    _, p_paired, _, _, _ = run_test(a, b, paired=True)
    assert p_paired < p_welch


def test_paired_uses_min_length():
    a = [10.0, 20.0, 30.0, 40.0]
    b = [9.0, 19.0]
    _, _, n_a, n_b, n_used = run_test(a, b, paired=True)
    assert (n_a, n_b, n_used) == (4, 2, 2)


def test_find_key_prefers_more_seeds():
    proto_small = (20, 5, 16)
    proto_big = (20, 5, 16)
    grouped = {
        ("folora_v2", "v2f_l300_k16", proto_small): {0: {}},
        ("folora_v2", "v2f_l300_k16", (10, 2, 8)): {0: {}, 1: {}, 2: {}},
    }
    # 同 tag 不同 protocol 时取 seed 数最多的
    assert find_key(grouped, "folora_v2", "v2f_l300_k16") == ("folora_v2", "v2f_l300_k16", (10, 2, 8))


def test_find_key_missing_returns_none():
    assert find_key({}, "ewc", "") is None


def test_compare_result_is_json_serializable():
    """scipy 返回 numpy 标量，直接塞进 dict 会让 json.dumps 炸掉。"""
    ours = ("folora_v2", "v2f_l300_k16", (20, 5, 16))
    base = ("ewc", "", (20, 5, 16))
    grouped = {
        ours: _seeds([0.10, 0.11, 0.09], [0.87, 0.88, 0.86]),
        base: _seeds([0.09, 0.08, 0.07], [0.89, 0.90, 0.88]),
    }
    r = compare(grouped, "cifar100", ours, base, "final_acc_cil", paired=False)
    assert isinstance(r["p"], float) and isinstance(r["significant"], bool)
    json.dumps(r)  # 不抛异常即通过
    assert r["delta"] > 0  # 本文方法准确率更高
