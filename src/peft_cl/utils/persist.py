"""checkpoint 的参数筛选规则。

**这里定义的是「哪些参数必须进 checkpoint」，它与「哪些参数可训练」是两件事。**
历史上这两件事被混为一谈（按 `requires_grad` 过滤），造成一个影响主表两行的缺陷：

    O-LoRA / InfLoRA 在 `before_task` 里把**历史任务的 adapter 冻结**
    （`olora.py` / `inflora.py` 的 `p.requires_grad = False`），
    于是任务 0..T-2 的 adapter 全部 `requires_grad=False`，
    按 `requires_grad` 过滤存盘时它们被整批丢掉 —— checkpoint 里只剩最后一个任务的 adapter。

    评估侧（`scripts/eval_ncm.py`）重建 adapter 时 `lora_B` 是零初始化
    （`multi_lora.py::_Adapter.__init__` 的 `nn.init.zeros_`），
    历史任务对 ΔW 的贡献恰好为 0，于是 Table 1 的 O-LoRA / InfLoRA 两行
    实际是用「只学了最后一个任务的模型」算出来的；断点续训同理会丢历史 adapter。

教训：**冻结参数一样可能是训练产物。** 判据必须建立在「这是不是预训练主干」上，
而不是「它还能不能训练」。

本模块独立成文件，是为了让单测能直接验证规则本身，而不必构造完整 CLTrainer
（后者要加载 CIFAR + ViT-B/16，建一个实例要几十秒）。
"""


def capture_base_params(model) -> dict:
    """在注入适配器**之前**调用：记下预训练主干的参数对象。

    存的是 `{id(param): param}` 而**不只是 id 集合**，因为我们持有对象引用可以防止
    对象被 GC 回收后其 id 被新张量复用 —— 那会让身份判据静默失效，表现是主干权重
    被误判为「新参数」而写进 checkpoint（每个涨到 ~400 MB），不会报错。
    """
    return {id(p): p for p in model.parameters()}


def is_persistent(param, base_params: dict) -> bool:
    """该参数是否需要进 checkpoint。两条判据取**并集**。

    1. `requires_grad` —— 正在训练的参数（分类头、prompt、单 LoRA、FOLoRA 核涉及的权重）；
    2. **不是**注入前就存在的预训练主干参数 —— 覆盖「被冻结、但属于方法状态」的参数，
       即 O-LoRA / InfLoRA 逐任务创建、随后被冻结的那些 adapter。

    判据 2 用**对象身份**而不是名字：注入 LoRA 时 `mlp.3` 被换成 LoRALinear，
    但原 `nn.Linear` 被持有为 `.base`，是同一个对象，所以 `...mlp.3.base.weight`
    仍会被正确地排除掉。

    被排除的正是冻结的预训练主干（ViT-B/16 约 86M 参数 ≈ 344 MB）——
    它们可由预训练权重重建，入库会让每个 checkpoint 从 ~50 MB 涨到 ~400 MB。
    """
    return bool(param.requires_grad) or base_params.get(id(param)) is not param


def persistent_model_state(model, base_params: dict) -> dict:
    """返回要写进 checkpoint 的 `{参数名: CPU 张量}`。"""
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if is_persistent(p, base_params)}


def persistent_param_names(model, base_params: dict) -> set:
    """同上，但只返回名字集合（供加载侧校验，不做 CPU 拷贝）。"""
    return {n for n, p in model.named_parameters() if is_persistent(p, base_params)}


def missing_persistent_keys(model, base_params: dict, model_state) -> set:
    """校验一份 model_state 是否覆盖了模型当前应有的全部持久化参数。

    返回**缺失的键名集合**；空集 = 该 checkpoint 是完整的。非空即说明它由旧版
    trainer 写出（按 requires_grad 过滤，冻结的历史 adapter 未入库）——
    加载它会让那些 adapter 保持随机初始化，且 `strict=False` **不会报任何错**。
    """
    return persistent_param_names(model, base_params) - set(model_state)
