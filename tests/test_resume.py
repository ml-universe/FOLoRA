"""断点续训正确性回归测试。

背景（曾踩坑）：O-LoRA / InfLoRA 的 adapter 是**逐任务动态创建**的，
而 trainer 灌 checkpoint 的顺序是「建模型 → load_state_dict(model_state)」。
若不先补建那些 adapter，`adapters.N.lora_A.weight` 这些键在模型里不存在，
会被 `strict=False` **静默丢弃** —— 表现是「续训后旧任务 adapter 全变随机初始化」，
结果错得极隐蔽。本文件锁住这个不变量。
"""

import copy

import torch
import torch.nn as nn

from peft_cl.adapters.multi_lora import inject_multi_lora
from peft_cl.methods.inflora import InfLoRAMethod
from peft_cl.methods.olora import OLoRAMethod
from peft_cl.utils.config import CLConfig


def _tiny_model():
    """构造一个含 `mlp.3` 的极小模型（inject_multi_lora 默认注入点名）。"""
    m = nn.Module()
    m.mlp = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                          nn.Linear(8, 4))
    return m


def _model_state(model):
    """模拟 trainer 落盘的可训参数集。"""
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad}


def _restore(model, state, last_task_id):
    """模拟 trainer 的续训流程，返回被丢弃的键名。"""
    own = dict(model.named_parameters())
    return [k for k in state if k not in own]


def test_dynamic_adapters_are_dropped_without_rebuild():
    """反例：不调用 rebuild_for_resume 时，checkpoint 的 adapter 键确实会丢失。

    这个测试固定「为什么需要 rebuild」这一事实；若哪天 trainer/加载顺序变了导致
    它不再成立，会提醒我们重新审视 rebuild 的必要性。
    """
    torch.manual_seed(0)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)

    model_a = _tiny_model()
    inject_multi_lora(model_a, cfg.lora_rank, cfg.lora_alpha)
    for ml in model_a.modules():
        if hasattr(ml, "adapters"):
            ml.add_task()
    for p in model_a.parameters():
        p.requires_grad = True
    state = _model_state(model_a)

    model_b = _tiny_model()
    inject_multi_lora(model_b, cfg.lora_rank, cfg.lora_alpha)
    dropped = _restore(model_b, state, 0)
    assert any("adapters.0.lora_A" in k for k in dropped), \
        "未重建时 adapter 键应被丢弃（这正是 bug 的成因）"


def test_olora_rebuild_restores_all_task_adapters():
    """正例：rebuild_for_resume + load_state_dict 后，所有任务的 adapter 权重一致。"""
    torch.manual_seed(0)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)

    model_a = _tiny_model()
    method_a = OLoRAMethod(model_a, cfg)
    for t in range(3):
        method_a.before_task(t)
        for p in model_a.parameters():
            p.requires_grad = True
        for p in model_a.parameters():
            with torch.no_grad():
                p.add_(torch.randn_like(p) * 0.01)
        method_a.after_task(t, None, None)
    saved_model = _model_state(model_a)
    saved_method = method_a.state_dict()

    # 全新进程：重建模型
    model_b = _tiny_model()
    method_b = OLoRAMethod(model_b, cfg)
    # 先重建动态结构，再灌权重（trainer 的正确顺序）
    method_b.rebuild_for_resume(2)
    dropped = _restore(model_b, saved_model, 2)
    assert dropped == [], f"重建后不应再有键被丢弃，实际丢弃: {dropped}"
    model_b.load_state_dict(saved_model, strict=False)
    method_b.load_state_dict(saved_method)

    own = dict(model_b.named_parameters())
    for name, tensor in saved_model.items():
        assert torch.equal(own[name], tensor), f"{name} 续训后不一致"
    assert len(method_b.prev_A[0]) == 3


def test_inflora_designed_basis_survives_roundtrip():
    """InfLoRA 设计出的降维基是冻结参数（不进 model_state），必须靠 method_state 恢复。"""
    torch.manual_seed(0)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)

    model_a = _tiny_model()
    method_a = InfLoRAMethod(model_a, cfg)
    method_a.rebuild_for_resume(2)                      # 3 个任务的分支
    designed = {}
    for i, ml in enumerate(method_a.multi_loras):
        method_a.designed_A[i] = []
        for j, ad in enumerate(ml.adapters):
            with torch.no_grad():
                ad.lora_A.weight.normal_()
            method_a.designed_A[i].append(ad.lora_A.weight.detach().clone())
        designed[i] = [a.clone() for a in method_a.designed_A[i]]
    saved_method = method_a.state_dict()

    # 注意：lora_A 是冻结的，不会出现在 model_state 里
    for p in model_a.parameters():
        p.requires_grad = True
    for ml in method_a.multi_loras:
        for ad in ml.adapters:
            ad.lora_A.weight.requires_grad = False
    saved_model = _model_state(model_a)
    assert not any("lora_A" in k for k in saved_model), \
        "lora_A 是冻结参数，不应出现在 model_state 中（故必须另行保存）"

    model_b = _tiny_model()
    method_b = InfLoRAMethod(model_b, cfg)
    method_b.rebuild_for_resume(2)
    model_b.load_state_dict(saved_model, strict=False)
    method_b.load_state_dict(saved_method)

    for i, ml in enumerate(method_b.multi_loras):
        for j, ad in enumerate(ml.adapters):
            assert torch.allclose(ad.lora_A.weight, designed[i][j]), \
                f"layer {i} task {j} 的设计基未恢复"
