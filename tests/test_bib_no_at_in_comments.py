"""护栏：refs.bib 在**条目之外**不得出现 `@`。

为什么值得一条独立测试
----------------------
BibTeX **没有注释字符**，也没有 `%`。解析器整个文件里只做一件事：找 `@`。
顶层任何位置的 `@` 都会让它开始解析一条条目，于是
`% 用 @misc 而非 @inproceedings：……`
这行会让 bibtex 报 `I was expecting a '{' or a '('`，**跳过那一条**，参考文献里
静默少一条，而 pdflatex 三次退出码全是 0。

这个坑在本项目已经发生过两次：内部决策日志记过一次，2026-10-04 我在新增
Group C 注释时**在修完上一次之后**又犯了一次。第二次能被立刻发现纯属运气——
碰巧检查了 bibtex 的退出码；如果哪次只跑 pdflatex、复用旧 `.bbl`，编译照样成功。

所以判据不是“编译通过”，而是“条目之外没有 `@`”。

判定口径（必须与 BibTeX 一致，过宽会误报）：逐个字符跟踪**是否位于某条目内部**
（用成对括号的嵌套深度判断），只有在条目**之外**看到 `@` 才算违规。
条目**内部**字段值里的 `@` 是合法的（邮箱、URL），不要报——把它们也报掉，这条护栏
迟早会被人关掉。
"""

import re
from pathlib import Path

import pytest

BIB = Path(__file__).resolve().parents[1] / "paper" / "refs.bib"

# 条目起始：行首（允许空白）+ @类型 + ( 或 {
ENTRY_START = re.compile(r"^\s*@[A-Za-z]+\s*([{(])")


def _offending_lines(text: str):
    """返回 [(行号, 行内容)]：位于条目之外的 `@`。"""
    out = []
    opener = closer = None          # 两者同时为 None ⇔ 当前不在条目内
    depth = 0
    for i, line in enumerate(text.splitlines(), start=1):
        if opener is None:
            m = ENTRY_START.match(line)
            if m:
                opener = m.group(1)
                closer = "}" if opener == "{" else ")"
                depth = line.count(opener) - line.count(closer)
                if depth <= 0:          # 单行条目，如 `@misc{k, title={x}}`
                    opener = closer = None
                    depth = 0
                continue
            if "@" in line:
                out.append((i, line))
            continue
        # 条目内：只跟踪嵌套深度；条目**内部**的 `@`（邮箱、URL）一律不管。
        # 用成对括号的深度而不是“下一个 @”来划边界，是因为字段值里就可能有 @。
        depth += line.count(opener) - line.count(closer)
        if depth <= 0:
            opener = closer = None
            depth = 0
    return out


def test_bib_exists():
    assert BIB.is_file(), f"找不到 {BIB}"


def test_no_at_outside_entries():
    bad = _offending_lines(BIB.read_text(encoding="utf-8"))
    if bad:
        shown = "\n".join(f"  refs.bib:{i}: {ln.strip()[:110]}" for i, ln in bad)
        pytest.fail(
            "refs.bib 在条目之外出现 `@` —— BibTeX 会把它当成新条目的开头，"
            "报 \"I was expecting a `{' or a `('\" 并**静默跳过条目**。\n"
            "注释里请写 `misc` / `inproceedings` 这类不带 `@` 的写法：\n" + shown)


def test_guard_actually_detects_the_regression():
    """反向自检：确认判据不是恒真（否则护栏本身是个摆设）。

    若这条失败，说明 `_offending_lines` 已经失效——通常是 `ENTRY_START` 写得太宽，
    把注释行也当成条目开头了，于是真回归会被默默放过。
    """
    bad = "  % 用 @misc 而非 @inproceedings：解释性注释\n@misc{real,\n  year = {2026}\n}\n"
    hits = _offending_lines(bad)
    assert len(hits) == 1 and hits[0][0] == 1, hits

    # 条目**内部**的 `@` 必须不报：这是防止护栏被“优化”成全文扫描的钉子。
    clean = "@misc{real,\n  note = {mail: a@b.com},\n  year = {2026}\n}\n"
    assert _offending_lines(clean) == []

    # 条目之外的普通文本行（没有 `%`）同样危险，必须一并抓到。
    assert len(_offending_lines("见 @liu2024x 的写法\n")) == 1
