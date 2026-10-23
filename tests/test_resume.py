"""断点续训 / checkpoint 持久化正确性回归测试。

本文件覆盖**两类**曾经踩过的坑，它们都表现为「静默丢失参数」：

1. **动态 adapter 的键对不上**：O-LoRA / InfLoRA 的 adapter 是逐任务动态创建的，
   trainer 灌 checkpoint 的顺序是「建模型 → load_state_dict(model_state)」。
   若不先 `rebuild_for_resume`，`adapters.N.lora_A.weight` 这些键在模型里不存在，
   会被 `strict=False` 静默丢弃。

2. **存盘时按 `requires_grad` 过滤**（2026-10-03 发现，影响主表两行）：
   O-LoRA / InfLoRA 的 `before_task` 把历史 adapter 全部**冻结**，于是任务 0..T-2 的
   adapter 全部 `requires_grad=False`，按 requires_grad 过滤会**整批丢掉**，
   checkpoint 里只剩最后一个任务的 adapter。评估侧重建 adapter 时 `lora_B` 是
   零初始化，历史任务对 ΔW 的贡献恰好为 0 —— 论文 Table 1 的 O-LoRA / InfLoRA 两行
   因此是「只学了最后一个任务的模型」的分数。

   **本文件原先把「按 requires_grad 过滤」手工写成 `_model_state()` 来模拟 trainer**，
   而那个模拟恰好复刻了缺陷本身，于是测试一路通过、缺陷照旧存在。
   现在改为直接调用生产实现 `peft_cl.utils.persist.persistent_model_state`。

教训：**测试不要复刻被测逻辑，要调用它。** 复刻出来的「模拟」会把 bug 一起复制进去，
然后通过。
"""

import torch
import torch.nn as nn

from peft_cl.adapters.multi_lora import inject_multi_lora
from peft_cl.methods.inflora import InfLoRAMethod
from peft_cl.methods.olora import OLoRAMethod
from peft_cl.utils.config import CLConfig
from peft_cl.utils.persist import (capture_base_params, is_persistent,
                                   missing_persistent_keys, persistent_model_state)


def _tiny_model():
    """构造一个含 `mlp.3` 的极小模型（inject_multi_lora 默认注入点名）。"""
    m = nn.Module()
    m.mlp = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                          nn.Linear(8, 4))
    return m


def _make_olora(freeze_history: bool, n_tasks: int = 3):
    """建一个 O-LoRA 模型，历史 adapter 的可训性按 freeze_history 决定。

    返回 (model, method, base_params)。base_params 在注入**之前**捕获，
    与 CLTrainer._build_model 的顺序一致。
    """
    torch.manual_seed(0)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)
    model = _tiny_model()
    base_params = capture_base_params(model)
    if freeze_history:
        method = OLoRAMethod(model, cfg)          # before_task 会冻结历史 adapter
        for t in range(n_tasks):
            method.before_task(t)
            for p in model.parameters():
                if p.requires_grad:
                    with torch.no_grad():
                        p.add_(torch.randn_like(p) * 0.01)
            method.after_task(t, None, None)
    else:
        # 对照组：手工把所有参数设成 requires_grad=True（这正是原来的测试做的事，
        # 也正是它掩盖缺陷的原因）
        method = OLoRAMethod(model, cfg)
        for t in range(n_tasks):
            method.before_task(t)
            for p in model.parameters():
                p.requires_grad = True
                with torch.no_grad():
                    p.add_(torch.randn_like(p) * 0.01)
            method.after_task(t, None, None)
    return model, method, base_params


# ---------------------------------------------------------------- 缺陷本体

def test_frozen_history_adapters_are_still_persisted():
    """**核心回归**：历史 adapter 被冻结时，它们仍必须进 checkpoint。

    这正是 2026-10-03 发现的主表缺陷：旧实现按 requires_grad 过滤，把
    tasks 0..T-2 的 adapter 整批丢掉，只留最后一个。
    """
    model, _method, base_params = _make_olora(freeze_history=True, n_tasks=3)
    state = persistent_model_state(model, base_params)

    # 前提：确实存在被冻结的历史 adapter（否则这个测试没测到东西）
    frozen = [n for n, p in model.named_parameters()
              if not p.requires_grad and ".adapters." in n]
    assert frozen, "构造失败：应当存在被冻结的历史 adapter"

    present = sorted({n.split(".adapters.")[1].split(".")[0] for n in state
                      if ".adapters." in n})
    assert present == ["0", "1", "2"], \
        f"历史 adapter 未被持久化（只存了 {present}）—— 这正是主表两行的缺陷"
    # 且被冻结的那些键必须在册
    assert all(n in state for n in frozen)


def test_old_requires_grad_filter_would_drop_them():
    """反例固定：按 requires_grad 过滤确实会丢历史 adapter（说明测试有分辨力）。"""
    model, _method, _base = _make_olora(freeze_history=True, n_tasks=3)
    buggy = {n for n, p in model.named_parameters() if p.requires_grad}
    present = sorted({n.split(".adapters.")[1].split(".")[0] for n in buggy
                      if ".adapters." in n})
    assert present == ["2"], \
        f"旧判据本应只留下最后一个 adapter，实际留下 {present}"


def test_base_backbone_params_are_excluded():
    """预训练主干（冻结、非方法新增）不进 checkpoint，否则每个文件涨到 ~400 MB。"""
    model, _method, base_params = _make_olora(freeze_history=True, n_tasks=2)
    state = persistent_model_state(model, base_params)
    assert not any(".base.weight" in n for n in state), \
        "冻结的主干权重被写进了 checkpoint（体积会暴涨）"
    # 而注入的 adapter 必须在册
    assert any(".adapters.0.lora_A.weight" in n for n in state)


def test_new_frozen_param_is_persistent():
    """判据的语义：只要不是注入前就有的主干参数，即使冻结也要持久化。"""
    model = _tiny_model()
    base_params = capture_base_params(model)
    extra = nn.Linear(4, 4, bias=False)
    extra.weight.requires_grad = False       # 冻结，但它是「新增」的
    model.add_module("extra", extra)
    assert is_persistent(extra.weight, base_params)
    # 主干自身的冻结参数则不是（必须先冻上，否则判据 1 会把它判成持久化）
    for p in model.mlp[3].parameters():
        p.requires_grad = False
    assert not is_persistent(next(iter(model.mlp[3].parameters())), base_params)


# ---------------------------------------------------------------- 续训流程

def test_dynamic_adapters_are_dropped_without_rebuild():
    """不调用 rebuild_for_resume 时，checkpoint 的 adapter 键确实会被丢弃。

    固定「为什么需要 rebuild」这一事实；若加载顺序变了导致它不再成立，
    会提醒我们重新审视 rebuild 的必要性。
    """
    torch.manual_seed(0)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)

    model_a = _tiny_model()
    base_a = capture_base_params(model_a)
    inject_multi_lora(model_a, cfg.lora_rank, cfg.lora_alpha)
    for ml in model_a.modules():
        if hasattr(ml, "adapters"):
            ml.add_task()
    state = persistent_model_state(model_a, base_a)

    model_b = _tiny_model()
    inject_multi_lora(model_b, cfg.lora_rank, cfg.lora_alpha)
    own = dict(model_b.named_parameters())
    dropped = [k for k in state if k not in own]
    assert any("adapters.0.lora_A" in k for k in dropped), \
        "未重建时 adapter 键应被丢弃（这正是 bug 的成因）"


def test_olora_resume_restores_every_task_adapter():
    """端到端：真实存盘规则 + rebuild_for_resume，所有任务的 adapter 一致。

    注意这里**不手工设置 requires_grad**（原测试的 `for p in ...: p.requires_grad = True`
    正是掩盖缺陷的那一行）。
    """
    model_a, method_a, base_a = _make_olora(freeze_history=True, n_tasks=3)
    saved_model = persistent_model_state(model_a, base_a)
    saved_method = method_a.state_dict()

    # 全新进程：重建模型（顺序与 trainer 一致：capture → 注入 → rebuild → 灌权重）
    torch.manual_seed(1)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)
    model_b = _tiny_model()
    base_b = capture_base_params(model_b)
    method_b = OLoRAMethod(model_b, cfg)
    method_b.rebuild_for_resume(2)

    assert missing_persistent_keys(model_b, base_b, saved_model) == set(), \
        "重建后不应再有应在册的参数缺失"
    model_b.load_state_dict(saved_model, strict=False)
    method_b.load_state_dict(saved_method)

    own = dict(model_b.named_parameters())
    for name, tensor in saved_model.items():
        assert torch.equal(own[name], tensor), f"{name} 续训后不一致"
    assert len(method_b.prev_A[0]) == 3
    # 历史 adapter 的 B 必须非零 —— 这才是「真的续训了」而不是「B 还是零初始化」
    for j in range(3):
        norm = float(saved_model[f"mlp.3.adapters.{j}.lora_B.weight"].norm())
        assert norm > 0, f"adapter {j} 的 lora_B 是零初始化（历史丢失）"


def test_inflora_designed_basis_survives_roundtrip():
    """InfLoRA 设计出的降维基必须能从 checkpoint 恢复。

    这些基是冻结参数。旧实现断言「lora_A 不应出现在 model_state 中」——
    那正是缺陷。现在它们既进 model_state，也由 method_state 冗余保存（向后兼容
    旧 checkpoint）。
    """
    torch.manual_seed(0)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)

    model_a = _tiny_model()
    base_a = capture_base_params(model_a)
    method_a = InfLoRAMethod(model_a, cfg)
    method_a.rebuild_for_resume(2)                      # 3 个任务的分支
    designed = {}
    for i, ml in enumerate(method_a.multi_loras):
        method_a.designed_A[i] = []
        for j, ad in enumerate(ml.adapters):
            with torch.no_grad():
                ad.lora_A.weight.normal_()
            ad.lora_A.weight.requires_grad = False     # 设计基是冻结的
            method_a.designed_A[i].append(ad.lora_A.weight.detach().clone())
        designed[i] = [a.clone() for a in method_a.designed_A[i]]
    saved_method = method_a.state_dict()
    saved_model = persistent_model_state(model_a, base_a)

    # **与旧断言相反**：冻结的设计基现在也必须进 model_state
    assert any("lora_A" in k for k in saved_model), \
        "冻结的设计基必须进 model_state（否则历史分支的 A 会丢失）"

    torch.manual_seed(1)
    model_b = _tiny_model()
    base_b = capture_base_params(model_b)
    method_b = InfLoRAMethod(model_b, cfg)
    method_b.rebuild_for_resume(2)
    assert missing_persistent_keys(model_b, base_b, saved_model) == set()
    model_b.load_state_dict(saved_model, strict=False)
    method_b.load_state_dict(saved_method)

    for i, ml in enumerate(method_b.multi_loras):
        for j, ad in enumerate(ml.adapters):
            assert torch.allclose(ad.lora_A.weight, designed[i][j]), \
                f"layer {i} task {j} 的设计基未恢复"


def test_inflora_history_lora_b_is_persisted():
    """InfLoRA 历史分支的 lora_B（可训练、训练后被冻结）必须持久化。

    旧实现只存了 `designed_A`（A 矩阵），历史分支的 B 完全没有持久化路径。
    """
    torch.manual_seed(0)
    cfg = CLConfig(lora_rank=2, lora_alpha=2)
    model = _tiny_model()
    base = capture_base_params(model)
    method = InfLoRAMethod(model, cfg)
    method.rebuild_for_resume(2)
    # 模拟训练：给历史分支的 B 写上非零值并冻结
    for ml in method.multi_loras:
        for ad in ml.adapters[:-1]:
            with torch.no_grad():
                ad.lora_B.weight.normal_()
            for p in ad.parameters():
                p.requires_grad = False
    state = persistent_model_state(model, base)
    for j in (0, 1):
        key = f"mlp.3.adapters.{j}.lora_B.weight"
        assert key in state, f"历史分支 {j} 的 lora_B 未持久化"
        assert float(state[key].norm()) > 0
