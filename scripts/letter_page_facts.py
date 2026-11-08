# -*- coding: utf-8 -*-
"""把回复信里那几句「几页、哪一节从第几页开始」的事实一次性算出来。

为什么要有这个脚本：页码不是常数。正文改三行、某张浮动表挪一页，附录的起始页就
跟着变，而回复信里写着 "body through p. 67, references pp. 67--81, appendix
pp. 81--90" 这种句子。手工每次去 PDF 里翻，既慢又容易看错一页；而**信里印错的
页码，审稿人是会拿 PDF 对着查的**。

做法：
* 节标题的页码从 `main.aux` 的 `\\newlabel` 读（唯一权威）；
* 「References」「Appendix A–D」这类没有 label 的标题，用 pdftotext 逐页扫首行；
* 正文结束页 = References 标题所在页（信里按「正文到此页为止」写）。

用法：
    py -3.11 -m scripts.letter_page_facts
"""

import re
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
AUX = PAPER / "main.aux"
PDF = PAPER / "main.pdf"

SECS = {
    "sec:intro": "§1",
    "sec:method": "§3",
    "sec:theory": "§4",
    "sec:experiments": "§5",
    "sec:conclusion": "§6",
}


def total_pages() -> int:
    txt = subprocess.run(["pdfinfo", str(PDF)], capture_output=True).stdout.decode(
        "utf-8", "replace")
    m = re.search(r"Pages:\s+(\d+)", txt)
    if m:
        return int(m.group(1))
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="ignore")
    return int(re.findall(r"Output written on .*?\((\d+) pages", log)[-1])


def aux_pages() -> dict:
    aux = AUX.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for key, name in SECS.items():
        m = re.search(r"\\newlabel\{" + re.escape(key) + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
        if m:
            out[name] = int(m.group(1))
    return out


def heading_pages() -> dict:
    """扫 PDF 每页，找没有 label 的标题。

    **不能只看首行**：正文长度一变，标题会从页首挪到页中，只看首行就会漏掉，
    然后误以为标题不存在。这里逐行扫，正文里偶尔出现的同名行由调用方按
    「第一次出现」处理——标题总在参考文献列表之前出现，所以取最早的一页是对的。
    """
    out = {}
    want = {"References": "exact", "Appendix A": "prefix", "Appendix B": "prefix",
            "Appendix C": "prefix", "Appendix D": "prefix"}
    for p in range(1, total_pages() + 1):
        raw = subprocess.run(
            ["pdftotext", "-f", str(p), "-l", str(p), "-layout", str(PDF), "-"],
            capture_output=True).stdout
        for line in raw.decode("utf-8", "replace").splitlines():
            s = line.strip()
            if not s:
                continue
            for key, how in want.items():
                if key in out:
                    continue
                hit = (s == key) if how == "exact" else s.startswith(key)
                if hit:
                    out[key] = p
    return out


def main():
    n = total_pages()
    secs = aux_pages()
    heads = heading_pages()

    # 附录 D 之前的最后一个标题就是附录起点；找不到就当正文结束页
    refs = heads.get("References")
    app = heads.get("Appendix A")
    print(f"总页数                 {n}")
    for name in ("§1", "§3", "§4", "§5", "§6"):
        if name in secs:
            print(f"{name} 起始页{'':<13}{secs[name]}")
    # 起止页都印成**不重叠**的区间：参考文献到附录起始页的前一页为止。
    # 旧版印成 refs--app 与 app--n，两个区间共用一页，照抄进回复信就成了自相矛盾的页码。
    if refs:
        print(f"References 起始页      {refs}   -> 正文 1--{refs - 1}")
    if app:
        print(f"Appendix A 起始页      {app}   -> 参考文献 {refs}--{app - 1}，附录 {app}--{n}")
    for k in ("Appendix B", "Appendix C", "Appendix D"):
        if k in heads:
            print(f"{k} 起始页{'':<11}{heads[k]}")
    if "§4" in secs and "§5" in secs:
        print(f"§4 跨度                {secs['§5'] - secs['§4']} 页")
    if "§5" in secs and "§6" in secs:
        print(f"§5 跨度                {secs['§6'] - secs['§5']} 页")


if __name__ == "__main__":
    main()
