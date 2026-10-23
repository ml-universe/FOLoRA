"""主干：冻结的 ViT + 可扩展分类头。"""
from .vit import ExpandableHead, build_vit

__all__ = ["build_vit", "ExpandableHead"]
