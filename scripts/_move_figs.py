# -*- coding: utf-8 -*-
"""一次性迁移：把 §5.2 的两张候选集曲线图移到附录（回应信中「已移到附录」的表述）。"""
from pathlib import Path

B = chr(92)  # 反斜杠，避免转义层级
SRC = Path("paper/sections/05_experiments.tex")
OUT = Path("paper/sections/_figs_candidate_curves.tex")

s = SRC.read_text(encoding="utf-8")
i = s.find("forgetting_curve_cifar100.pdf")
assert i > 0, "找不到第一张图的 includegraphics"
start = s.rindex(B + "begin{figure}", 0, i)
end = s.find(B + "subsection{How much of the ranking", i)
assert end > start, (start, end)
block = s[start:end]
assert block.count(B + "begin{figure}") == 2, block.count(B + "begin{figure}")
assert block.count(B + "end{figure}") == 2

SRC.write_text(s[:start] + s[end:], encoding="utf-8")
OUT.write_text(block.rstrip() + "\n", encoding="utf-8")
print("moved", len(block), "chars:", start, "->", end)
