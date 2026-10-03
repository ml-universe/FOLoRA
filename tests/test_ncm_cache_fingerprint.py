# -*- coding: utf-8 -*-
"""NCM 逐 run 评估缓存的内容指纹（P2-5）。

背景：`reports/ncm/` 下的逐 run 缓存原本「文件存在即跳过」。缓存名里**含 tag**，
所以正常重训（换 tag）本就会换缓存键、强制重算；唯一的漏洞是**同一个 tag 原地覆盖**
—— 那时键不变，评估结果会永远停留旧实现上**且没有任何提示**。这正是 P0-1 那一类
「评估的模型 ≠ 训练的模型」的静默错，只是发生在缓存层。

修法刻意**不用 mtime**（拷贝/备份还原/git checkout 会改它，多进程与时钟漂移下也
不可预测 —— 这条顾虑写在 eval_ncm_sweep 的模块 docstring 里），改用内容派生的
`eval_ncm.source_fingerprint`。本文件把两条性质都钉住：

  1. 指纹本身：稳定（同文件同指纹）、灵敏（size 或尾部变则变）；
  2. 决策路径：指纹不符 -> 重算；旧缓存无该字段 -> 沿用并提示（不作废历史缓存）。

全程不加载模型、不碰 GPU：evaluate_one 被替换成假实现。
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout

import pytest

import scripts.eval_ncm_sweep as sweep
from scripts.eval_ncm import source_fingerprint


# --------------------------------------------------------------- 指纹本体

def test_fingerprint_is_stable_for_same_file(tmp_path):
    p = tmp_path / "checkpoint.pt"
    p.write_bytes(b"A" * 4096)
    assert source_fingerprint(tmp_path) == source_fingerprint(tmp_path)


def test_fingerprint_none_without_checkpoint(tmp_path):
    """SimpleCIL 的冻结特征缓存没有 checkpoint —— 必须返回 None（调用方据此
    视为「无法判断」而不是「已过期」）。"""
    assert source_fingerprint(tmp_path) is None


def test_fingerprint_detects_size_and_tail_change(tmp_path):
    p = tmp_path / "checkpoint.pt"
    p.write_bytes(b"A" * (3 << 20))            # 3 MiB，大于取样块 2×1 MiB
    f0 = source_fingerprint(tmp_path)

    with open(p, "r+b") as f:                  # 尾部改一字节
        f.seek(-1, os.SEEK_END)
        f.write(b"B")
    f1 = source_fingerprint(tmp_path)
    assert f0 != f1, "尾部变了指纹却没变"

    with open(p, "ab") as f:                   # 追加 -> size 变
        f.write(b"C")
    f2 = source_fingerprint(tmp_path)
    assert f1 != f2 and f2["size"] == f1["size"] + 1, "size 变了指纹却没变"


def test_fingerprint_detects_small_file_change(tmp_path):
    """小于取样块的文件：头块即全文，内容变了必须能检出。"""
    p = tmp_path / "checkpoint.pt"
    p.write_bytes(b"X" * 100)
    f0 = source_fingerprint(tmp_path)
    p.write_bytes(b"Y" * 100)
    f1 = source_fingerprint(tmp_path)
    assert f0 != f1 and f0["size"] == f1["size"]


# --------------------------------------------------------- 缓存决策路径

@pytest.fixture
def bench(tmp_path, monkeypatch):
    """造一个最小 experiments 树，并把 evaluate_one 换成记录调用的假实现。"""
    calls = []

    def fake_evaluate_one(benchmark, num_tasks, seed, device, **kw):
        calls.append(kw.get("run_dir"))
        return {"final_acc_cil": 0.75, "forgetting_cil": 0.10,
                "incremental_acc_cil": 0.60, "acc_cil": [[0.7] * 20] * 20,
                "load_audit": {"n_unmatched": 0, "n_shape_mismatch": 0}}

    monkeypatch.setattr(sweep, "evaluate_one", fake_evaluate_one)

    exp = tmp_path / "experiments"
    cache = tmp_path / "reports" / "ncm"
    run = exp / "cifar100" / "olora" / "default" / "seed0"
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({
        "finished": True, "config": {"num_tasks": 20},
        "final_acc_til": 0.9, "forgetting_til": 0.05,
        "final_acc_cil": 0.8, "forgetting_cil": 0.2}), encoding="utf-8")
    (run / "checkpoint.pt").write_bytes(b"A" * (3 << 20))

    def run_sweep():
        calls.clear()
        monkeypatch.setattr(sys, "argv", [
            "eval_ncm_sweep", "--benchmark", "cifar100", "--methods", "olora",
            "--exp_root", str(exp), "--cache_root", str(cache),
            "--out", str(tmp_path / "summary.json")])
        buf = io.StringIO()
        with redirect_stdout(buf):
            sweep.main()
        return buf.getvalue()

    return {"run": run, "cache": cache, "calls": calls, "sweep": run_sweep}


def _cache_file(bench, tag="default"):
    return bench["cache"] / "cifar100" / f"olora__{tag}__seed0.json"


def test_writes_fingerprint_then_hits_cache(bench):
    out = bench["sweep"]()
    # 首轮除本 run 外还会顺带算一次冻结特征 SimpleCIL（run_dir=None）
    assert len(bench["calls"]) == 2, bench["calls"]
    rec = json.loads(_cache_file(bench).read_text(encoding="utf-8"))
    assert rec["src_fp"] == source_fingerprint(bench["run"])

    out = bench["sweep"]()
    assert bench["calls"] == [], "指纹未变却重新评估了"
    assert "[cached]" in out


def test_recomputes_when_fingerprint_mismatches(bench):
    """模拟同一 tag 原地覆盖：缓存里的指纹与实际 checkpoint 不符。"""
    bench["sweep"]()
    cp = _cache_file(bench)
    rec = json.loads(cp.read_text(encoding="utf-8"))
    rec["src_fp"] = {"size": 1, "sha256_ht": "0" * 32}
    cp.write_text(json.dumps(rec), encoding="utf-8")

    out = bench["sweep"]()
    assert len(bench["calls"]) == 1, "指纹不符却没有重算"
    assert "[stale]" in out
    assert json.loads(cp.read_text(encoding="utf-8"))["src_fp"] == \
        source_fingerprint(bench["run"])


def test_recomputes_when_checkpoint_changes(bench):
    bench["sweep"]()
    ckpt = bench["run"] / "checkpoint.pt"
    with open(ckpt, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        f.write(b"B")

    out = bench["sweep"]()
    assert len(bench["calls"]) == 1 and "[stale]" in out


def test_legacy_cache_without_fingerprint_is_kept(bench):
    """旧缓存没有 src_fp：必须沿用（否则会一举作废全部历史缓存、触发整套 GPU
    重评估），但要打印提示让重训者知道内容无法核对。"""
    bench["sweep"]()
    cp = _cache_file(bench)
    rec = json.loads(cp.read_text(encoding="utf-8"))
    rec.pop("src_fp")
    cp.write_text(json.dumps(rec), encoding="utf-8")

    out = bench["sweep"]()
    assert bench["calls"] == [], "旧缓存被作废重算了"
    assert "[note ]" in out and "[cached]" in out


def test_new_tag_gets_its_own_cache_entry(bench):
    """本仓库铁律路径：重训写新 tag -> 缓存键不同 -> 必然评估。"""
    bench["sweep"]()
    src = bench["run"]
    run2 = src.parent.parent / "default_fix1" / "seed0"
    run2.mkdir(parents=True)
    (run2 / "results.json").write_text(
        (src / "results.json").read_text(encoding="utf-8"), encoding="utf-8")
    (run2 / "checkpoint.pt").write_bytes(b"A" * (3 << 20))

    bench["sweep"]()
    assert len(bench["calls"]) == 1, bench["calls"]
    assert _cache_file(bench, "default").exists()
    assert _cache_file(bench, "default_fix1").exists()
