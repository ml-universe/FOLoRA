"""checkpoint 审计与加载守卫的测试。

背景：2026-10-03 发现 O-LoRA / InfLoRA 的 checkpoint 只含最后一个任务的 adapter
（trainer 按 `requires_grad` 过滤存盘，而历史 adapter 被冻结）。这个缺陷之所以能长期
存在，是因为**评估侧只看 print 出来的命中数、既不记录也不校验**，`strict=False`
也不会报错。本文件锁住新加的两道防线：

1. 加载守卫 `persist.missing_persistent_keys` —— 续训前硬失败；
2. 评估审计 `eval_ncm.audit_adapter_coverage` —— 把「缺哪些任务的 adapter」显式报出来，
   并随结果落盘。
"""

import torch
import torch.nn as nn

from peft_cl.adapters.multi_lora import inject_multi_lora
from peft_cl.utils.config import CLConfig
from peft_cl.utils.persist import capture_base_params, missing_persistent_keys, \
    persistent_model_state
from scripts.eval_ncm import audit_adapter_coverage


def _tiny_model():
    m = nn.Module()
    m.mlp = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                          nn.Linear(8, 4))
    return m


def _multi_lora_model(frozen_history: bool, n_tasks: int = 4):
    """建一个 n_tasks 个 adapter 的模型；frozen_history=True 时冻结除最后一个之外全部。"""
    torch.manual_seed(0)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)
    model = _tiny_model()
    base = capture_base_params(model)
    inject_multi_lora(model, cfg.lora_rank, cfg.lora_alpha)
    for ml in model.modules():
        if hasattr(ml, "adapters"):
            for t in range(n_tasks):
                ad = ml.add_task()
                with torch.no_grad():
                    ad.lora_B.weight.normal_()          # 模拟「训练过」
                if frozen_history and t < n_tasks - 1:
                    for p in ad.parameters():
                        p.requires_grad = False
    return model, base


# ---------------------------------------------------------------- 评估侧审计

def test_audit_flags_checkpoint_missing_history_adapters():
    """只有最后一个 adapter 的 checkpoint 必须被识别出来（并列出缺失的任务号）。"""
    model, base = _multi_lora_model(frozen_history=True, n_tasks=4)
    # 复刻旧版 trainer 的 buggy 存盘：按 requires_grad 过滤
    buggy_state = {n: p.detach().cpu() for n, p in model.named_parameters()
                   if p.requires_grad}
    present, missing = audit_adapter_coverage(buggy_state, num_tasks=4)
    assert present == {3}, f"旧判据本应只留下任务 3，实际 {present}"
    assert missing == [0, 1, 2], f"缺失任务号应为 [0,1,2]，实际 {missing}"


def test_audit_passes_on_complete_checkpoint():
    """修复后的存盘规则必须让审计通过。"""
    model, base = _multi_lora_model(frozen_history=True, n_tasks=4)
    good_state = persistent_model_state(model, base)
    present, missing = audit_adapter_coverage(good_state, num_tasks=4)
    assert present == {0, 1, 2, 3}
    assert missing == []


def test_audit_ignores_non_adapter_methods():
    """单 LoRA 的方法（seq/ewc/folora）没有 .adapters.N. 键，审计不应误报。"""
    state = {"encoder.layers.encoder_layer_0.mlp.3.lora_A.weight": torch.zeros(2, 8),
             "heads.head.weight": torch.zeros(4, 8)}
    present, missing = audit_adapter_coverage(state, num_tasks=20)
    assert present == set()
    # 注意：这里 missing 会列出全部 20 个任务，但调用方（evaluate_one）只在
    # config.method 使用逐任务 adapter 时才看这个字段——见其注释。
    assert missing == list(range(20))


# ---------------------------------------------------------------- 加载守卫

def test_load_guard_raises_on_buggy_checkpoint():
    """续训守卫：缺陷 checkpoint 必须硬失败，而不是静默带着随机 adapter 继续训练。"""
    model_a, base_a = _multi_lora_model(frozen_history=True, n_tasks=4)
    buggy = {n: p.detach().cpu() for n, p in model_a.named_parameters()
             if p.requires_grad}

    # 新进程：结构重建好了，但 checkpoint 是残缺的
    model_b, base_b = _multi_lora_model(frozen_history=False, n_tasks=4)
    missing = missing_persistent_keys(model_b, base_b, buggy)
    assert missing, "残缺 checkpoint 必须被守卫识别出来"
    assert any(".adapters.0." in k for k in missing)
    assert not any(".adapters.3." in k for k in missing)


def test_load_guard_passes_on_complete_checkpoint():
    model_a, base_a = _multi_lora_model(frozen_history=True, n_tasks=4)
    good = persistent_model_state(model_a, base_a)
    model_b, base_b = _multi_lora_model(frozen_history=False, n_tasks=4)
    assert missing_persistent_keys(model_b, base_b, good) == set()
