"""持续学习方法：seq / ewc / olora / inflora / folora / folora_v2 / l2p / coda。"""
from .base import CLMethod
from .seq import SeqMethod
from .ewc import EWCMethod
from .olora import OLoRAMethod
from .inflora import InfLoRAMethod
from .folora import FOLoRAMethod
from .folora_v2 import FOLoRAv2Method
from .l2p import L2PMethod
from .coda import CODAMethod


def build_method(name: str, model, config) -> CLMethod:
    """按名字构造方法实例。

    `"folora"` 是 v1 的历史别名。v1 的正则项是退化实现（见 `methods/folora.py` 的
    docstring），用它跑实验会得到「无正交惩罚」的结果却不报错——论文的方法是
    `folora_v2`。这里保留该名字仅为读旧 checkpoint，并显式告警。
    """
    registry = {
        "seq": SeqMethod,
        "ewc": EWCMethod,
        "olora": OLoRAMethod,
        "inflora": InfLoRAMethod,
        "folora": FOLoRAMethod,
        "folora_v2": FOLoRAv2Method,
        "l2p": L2PMethod,
        "coda": CODAMethod,
    }
    if name not in registry:
        raise ValueError(f"unknown method: {name}")
    if name == "folora":
        import warnings
        warnings.warn(
            "method 'folora' is the legacy v1 (degenerate regularizer). "
            "The paper's method is 'folora_v2'; use 'folora' only to load old checkpoints.",
            DeprecationWarning, stacklevel=2)
    return registry[name](model, config)
