"""原子 JSON 读写（断电续训的基础设施）。

为什么需要
----------
所有「断点续跑」的调度都靠「文件存在且内容完整 = 这个单元已完成」来判断。
如果写盘途中断电，会得到一个**被截断的 JSON**，后果分两种，都很糟：

- 调度器 `json.loads` 抛异常 → 整个队列崩掉，后面所有实验都不跑；
- 或者文件恰好能被解析但内容不全 → 该单元被误判为「已完成」，结果永久错误。

因此约定：
- 写：先写同目录的 `.tmp`，**fsync 落盘**后再 `os.replace` 原子替换。这样任何时刻
  目标文件要么是旧的完整内容、要么是新的完整内容，不存在中间态。
- 读：任何失败（不存在／截断／被占用）都返回 None，让调用方把该单元当成「未完成」
  重跑，而不是让整个队列崩掉。

注意 fsync 不能省：只做 `os.replace` 而不 fsync 时，数据可能仍在页缓存里，
断电后目标文件依然是空的或旧的——「原子替换」保护的是名字，不是内容。
"""

import json
import os
from pathlib import Path


def atomic_write_json(path, obj, indent: int = 2) -> None:
    """把 obj 原子地写成 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path, data: bytes) -> None:
    """原子地写二进制文件（例如 torch.save 出来的 checkpoint）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json_or_none(path):
    """读 JSON；不存在／被截断／被占用时返回 None（调用方据此重跑该单元）。"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
