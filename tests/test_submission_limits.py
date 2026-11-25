"""护栏：摘要与 highlights 必须落在期刊的硬上限内。

为什么值得一条测试
------------------
两个上限不是「风格建议」，是 EM/出版社技术审查会直接退的东西：

* 摘要 ≤ 250 词（Neuroscience 系期刊通用；第一轮送审版 248 词，是压着线写的）；
* highlights 3–5 条，每条 ≤ 85 字符（含空格）。

第二轮改摘要时（R8-2 加三条局限、R8-3 改归因）摘要一度涨到 375 词、两条 highlights
到 103/105 字符——正文和 PDF 都正常编译，没有任何测试会红，只有交上去才被退。
所以这里按 `make_submission_docs` 真正写进 .docx 的那份扁平文本量，而不是数源码里的
宏：docx 是 EM 实际收的东西。
"""

import pytest

from scripts.make_submission_docs import read_blocks

ABSTRACT_MAX_WORDS = 250
HIGHLIGHT_MAX_CHARS = 85
HIGHLIGHT_MIN, HIGHLIGHT_MAX = 3, 5


@pytest.fixture(scope="module")
def blocks():
    return read_blocks()


def test_abstract_within_word_limit(blocks):
    abstract, _ = blocks
    n = len(abstract.split())
    assert n <= ABSTRACT_MAX_WORDS, (
        f"摘要 {n} 词，超过上限 {ABSTRACT_MAX_WORDS}（docx 里就是这段文字）")


def test_highlight_count(blocks):
    _, items = blocks
    assert HIGHLIGHT_MIN <= len(items) <= HIGHLIGHT_MAX, \
        f"highlights {len(items)} 条，应为 {HIGHLIGHT_MIN}–{HIGHLIGHT_MAX} 条"


def test_each_highlight_within_char_limit(blocks):
    _, items = blocks
    over = [(len(x), x) for x in items if len(x) > HIGHLIGHT_MAX_CHARS]
    assert not over, "超 85 字符的 highlights：\n" + "\n".join(
        f"  {n} | {x}" for n, x in over)
