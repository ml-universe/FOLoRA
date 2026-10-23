"""让 tests/ 既能 `python -m pytest` 也能裸 `pytest` 跑。

peft_cl 本身是 `pip install -e .` 装好的；这里的 sys.path 只为 scripts/ 下的
分析脚本（不是安装包的一部分）能被测试导入。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
