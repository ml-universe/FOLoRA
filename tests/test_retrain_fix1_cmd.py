# -*- coding: utf-8 -*-
"""回归：重训命令必须被 `run_single` 的 parser 接受，且 `--folora_weighted` 要带值。

背景（2026-10-04 事故）
----------------------
`retrain_fix1.build_cmd` 原先靠 `isinstance(v, bool)` 判断「配置里是布尔 ⇒ CLI 是
store_true」，但 `--folora_weighted` 是**例外**：配置里是 bool，CLI 却是
`type=int, choices=[0,1]`（必须带值）。于是 22 个源 config 里记了
`folora_weighted: true` 的 run 被生成成裸 `--folora_weighted`，argparse 报
`expected one argument` 后 exit 2，**0.0 分钟秒退**；另外 13 个源 config 没记这个字段，
命令里根本没它，就正常跑完了 —— 一批里半成半败、失败者都"秒退"，就是这么来的。

修法：不再从**值的类型**反推 CLI 形状，改为直接问 CLI（`build_parser()` 是唯一事实来源）。
本文件把这条钉死。
"""
from scripts.retrain_fix1 import STORE_TRUE_OPTIONS, build_cmd
from scripts.run_single import build_parser


def _parse(cmd):
    """复刻 retrain_fix1 的调用方式：cmd 前三个元素是 [python, -m, scripts.run_single]。"""
    assert cmd[:3] == [cmd[0], "-m", "scripts.run_single"]
    return build_parser().parse_args(cmd[3:])


def test_folora_weighted_is_not_a_store_true_option():
    """它是 int/choices 选项 —— 这正是当初判断反了的那一个。"""
    assert "folora_weighted" not in STORE_TRUE_OPTIONS
    assert "bound_probe" in STORE_TRUE_OPTIONS


def test_weighted_true_emits_a_value():
    cmd = build_cmd({"folora_weighted": True, "benchmark": "cifar100"}, "t_fix1")
    i = cmd.index("--folora_weighted")
    assert cmd[i + 1] == "1", f"--folora_weighted 必须带值，实际命令: {cmd}"
    assert _parse(cmd).folora_weighted == 1


def test_weighted_false_emits_zero_not_a_bare_flag():
    cmd = build_cmd({"folora_weighted": False}, "t_fix1")
    i = cmd.index("--folora_weighted")
    assert cmd[i + 1] == "0"
    assert _parse(cmd).folora_weighted == 0


def test_absent_key_omits_the_flag():
    """源 config 没记这个字段时，命令里就不该出现它（走 CLI 默认 1）。"""
    cmd = build_cmd({"benchmark": "cifar100"}, "t_fix1")
    assert "--folora_weighted" not in cmd
    assert _parse(cmd).folora_weighted == 1


def test_bound_probe_true_is_a_bare_flag():
    """真正的 store_true 仍然发裸 flag —— 修复不能把这类一起改坏。"""
    cmd = build_cmd({"bound_probe": True}, "t_fix1")
    assert "--bound_probe" in cmd
    assert _parse(cmd).bound_probe is True


def test_bound_probe_false_omits_the_flag():
    cmd = build_cmd({"bound_probe": False}, "t_fix1")
    assert "--bound_probe" not in cmd


def test_every_cli_exposed_field_replays():
    """全字段非默认的 config 走一遍 build_cmd + parse，逐字段必须一致。"""
    src = {
        "benchmark": "imagenetr", "num_tasks": 20, "method": "olora", "seed": 3,
        "epochs": 5, "batch_size": 32, "lr": 1e-3, "lora_rank": 16, "lora_alpha": 16,
        "folora_lambda": 300.0, "folora_topk": 0, "folora_weighted": True,
        "olora_aggregate": "sum", "olora_orth_lambda": 0.1, "ewc_lambda": 100.0,
        "prompt_pool_size": 20, "prompt_length": 5, "prompt_topk": 5,
        "fisher_batches": 300, "data_root": "data", "out_dir": "experiments",
    }
    a = _parse(build_cmd(src, "t_fix1"))
    for f in ("benchmark", "num_tasks", "method", "seed", "epochs", "batch_size", "lr",
              "lora_rank", "lora_alpha", "folora_lambda", "folora_topk",
              "olora_aggregate", "olora_orth_lambda", "ewc_lambda", "prompt_pool_size",
              "prompt_length", "prompt_topk", "fisher_batches", "data_root", "out_dir"):
        assert getattr(a, f) == src[f], f
    assert a.folora_weighted == 1


def test_tag_is_appended_last():
    cmd = build_cmd({"benchmark": "cifar100"}, "olora_orth_l1_fix1")
    assert cmd[-2:] == ["--tag", "olora_orth_l1_fix1"]


def test_emoji_print_survives_gbk_redirect(tmp_path):
    """回归（2026-10-04）：stdout 重定向到文件时打印 ✅ 不能把脚本打挂。

    那次：第一个 run 跑完 65 分钟、results.json 已落盘，却卡在打印 ✅ 那行
    `UnicodeEncodeError`（Windows 重定向到文件时按系统区域编码走 GBK）退出，
    整脚本死掉、白等一小时。这里在**不带 PYTHONIOENCODING** 的子进程里复现那个环境。
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from scripts.retrain_fix1 import _force_utf8_stdout as f;"
        "f(); print('\\u2705 \\u274c ok')" % str(root)
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    out = tmp_path / "redirected.txt"
    with open(out, "wb") as fh:
        r = subprocess.run([sys.executable, "-c", code], stdout=fh,
                           stderr=subprocess.PIPE, env=env, cwd=str(root))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "✅" in out.read_text(encoding="utf-8")
