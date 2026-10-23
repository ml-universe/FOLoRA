"""日志：同时写控制台 + UTF-8 文件（文件用 append，支持断点续训叠加日志）。

注意：控制台消息刻意用英文——Windows 控制台是 GBK 编码，打印中文会乱码；
文件 handler 显式 UTF-8，中文 docstring/注释在 PyCharm 里正常显示不受影响。
"""

import logging
import sys
from pathlib import Path


def setup_logger(log_file: str, name: str = "peft_cl") -> logging.Logger:
    """返回一个同时输出到控制台与文件的 logger（文件追加写，续训不丢历史）。"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    # 幂等：重复 setup 不叠加 handler
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
