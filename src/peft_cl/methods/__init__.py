"""持续学习方法：seq / ewc / olora / folora / folora_v2 / l2p / coda。"""
from .base import CLMethod
from .seq import SeqMethod
from .ewc import EWCMethod
from .olora import OLoRAMethod
from .folora import FOLoRAMethod
from .folora_v2 import FOLoRAv2Method
from .l2p import L2PMethod
from .coda import CODAMethod


def build_method(name: str, model, config) -> CLMethod:
    """按名字构造方法实例。"""
    registry = {
        "seq": SeqMethod,
        "ewc": EWCMethod,
        "olora": OLoRAMethod,
        "folora": FOLoRAMethod,
        "folora_v2": FOLoRAv2Method,
        "l2p": L2PMethod,
        "coda": CODAMethod,
    }
    if name not in registry:
        raise ValueError(f"unknown method: {name}")
    return registry[name](model, config)
