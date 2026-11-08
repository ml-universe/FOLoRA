#!/usr/bin/env python3
"""Render paper/response-letter/ as plain ASCII text for pasting into EM.

The PDF (paper/response-letter/main.pdf) is the authoritative response; this
script exists because Editorial Manager sometimes asks for the response in a
text field, where a LaTeX table pastes as an unreadable column of fragments.

It reads the same .tex sources as the PDF, so every number in the text is the
same number that was typeset, and it lowers each table to either a fixed-width
pipe table (numeric rows) or a labelled list (wide prose rows).

Run from the repository root:

    py -3.11 scripts/make_response_plaintext.py --out <directory>
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

LETTER_FILES = [
    "01_summary.tex",
    "02_reviewer1.tex",
    "03_reviewer3.tex",
    "04_reviewer5.tex",
    "05_reviewer7.tex",
    "06_reviewer8.tex",
    "07_files.tex",
]

WIDTH = 88
# Keep words like "Seq-LoRA" and "R7-W3" in one piece when wrapping.
WRAP_KW = {"break_on_hyphens": False}

GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "varepsilon": "epsilon", "zeta": "zeta", "eta": "eta",
    "theta": "theta", "kappa": "kappa", "lambda": "lambda", "mu": "mu",
    "nu": "nu", "xi": "xi", "rho": "rho", "sigma": "sigma", "tau": "tau",
    "phi": "phi", "varphi": "phi", "chi": "chi", "psi": "psi", "omega": "omega",
    "Delta": "Delta", "Sigma": "Sigma", "Theta": "Theta", "Lambda": "Lambda",
    "Omega": "Omega", "Gamma": "Gamma", "Phi": "Phi", "Psi": "Psi",
}

MATH_OPS = {
    "le": "<=", "leq": "<=", "ge": ">=", "geq": ">=", "neq": "!=",
    "equiv": "=", "ll": "<<", "gg": ">>", "propto": "~", "ast": "*",
    "approx": "~", "sim": "~", "simeq": "~", "cong": "~",
    "pm": "+/-", "mp": "-/+", "times": "x", "cdot": " * ", "dots": "...",
    "ldots": "...", "cdots": "...", "vdots": "...",
    "to": "->", "rightarrow": "->", "Rightarrow": "=>", "implies": "=>",
    "in": " in ", "infty": "inf", "partial": "d", "mid": "|",
    "lVert": "||", "rVert": "||", "vert": "|", "lvert": "|", "rvert": "|",
    "top": "T ", "ell": "l", "nabla": "grad", "forall": "for all",
    "exists": "exists", "propto": "~", "subset": "subset",
}

NAME_TEXT = {"LaTeX": "LaTeX", "TeX": "TeX", "BibTeX": "BibTeX", "XeTeX": "XeTeX"}

DROP = {
    "noindent", "centering", "small", "footnotesize", "scriptsize", "tiny",
    "normalsize", "large", "Large", "LARGE", "huge", "Huge", "bfseries",
    "itshape", "ttfamily", "rmfamily", "sffamily", "upshape", "slshape",
    "scshape", "hrule", "par", "smallskip", "medskip", "bigskip", "vfill",
    "hfill", "clearpage", "newpage", "allowbreak", "relax", "leavevmode",
    "displaybreak", "protect", "strut", "phantomsection", "raggedright",
    "arraybackslash", "centering",
}

WRAP_CMDS = {
    "emph", "textbf", "textit", "texttt", "textrm", "textsf", "textsc",
    "textsl", "text", "mathrm", "mathit", "mathbf", "mathsf", "mathtt",
    "operatorname", "mbox", "underline", "ensuremath", "textup",
}

SYMBOL = {
    "%": "%", "&": "&", "_": "_", "#": "#", "{": "{", "}": "}", "$": "$",
    " ": " ", ",": " ", ";": " ", ":": " ", "!": "", "-": "", "|": "||",
    "/": "/", "\\": "\n",
}

ROMAN = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
         "xi", "xii"]


def match_brace(s: str, i: int) -> int:
    """Index of the '}' matching the '{' at s[i], skipping escaped braces."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError(f"unbalanced braces at {i}: {s[i:i + 60]!r}")


def take_arg(s: str, i: int) -> tuple[str, int]:
    """Read a braced argument at s[i], or a single token if unbraced."""
    if i < len(s) and s[i] == "{":
        j = match_brace(s, i)
        return s[i + 1:j], j + 1
    if i < len(s):
        return s[i], i + 1
    return "", i


def parse(s: str, math: bool = False) -> str:
    """Tokenise a fragment of LaTeX into ASCII text."""
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            if s.startswith("\\[", i):
                j = s.find("\\]", i + 2)
                if j != -1:
                    out.append(space_greek(parse(s[i + 2:j], True)))
                    i = j + 2
                    continue
            if i + 1 < n and s[i + 1] == "\\":
                j = i + 2
                if j < n and s[j] == "[":
                    j = s.index("]", j) + 1
                out.append("\n")
                i = j
                continue
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if not m:
                sym = s[i + 1] if i + 1 < n else ""
                out.append(SYMBOL.get(sym, sym))
                i += 2
                continue
            name = m.group(1)
            i += m.end()

            if name in ("frac", "tfrac", "dfrac"):
                a, i = take_arg(s, i)
                b, i = take_arg(s, i)
                out.append(parse(a, True) + "/" + parse(b, True))
                continue
            if name in WRAP_CMDS:
                arg, i = take_arg(s, i)
                out.append(parse(arg, math))
                continue
            if name == "href":
                url, i = take_arg(s, i)
                text, i = take_arg(s, i)
                out.append(parse(text, math) or url)
                continue
            if name == "url":
                arg, i = take_arg(s, i)
                out.append(arg)
                continue
            if name in ("bar", "hat", "tilde", "vec", "overline"):
                while i < len(s) and s[i] == " ":
                    i += 1
                arg, i = take_arg(s, i)
                tag = "bar" if name in ("bar", "overline") else name
                out.append(parse(arg, math) + "_" + tag)
                continue
            if name == "sqrt":
                arg, i = take_arg(s, i)
                out.append("sqrt(" + parse(arg, True) + ")")
                continue
            if name == "sum":
                if out and out[-1] and (out[-1][-1].isalnum() or out[-1][-1] == "_"):
                    out.append(" ")
                if i < len(s) and s[i] == "_":
                    arg, i = take_arg(s, i + 1)
                    out.append("sum over " + parse(arg, True) + " ")
                else:
                    out.append("sum ")
                continue
            if name in GREEK:
                out.append(GREEK[name])
                continue
            if name in MATH_OPS:
                out.append(MATH_OPS[name])
                continue
            if name in NAME_TEXT:
                if s[i:i + 2] == "{}":
                    i += 2
                out.append(NAME_TEXT[name])
                continue
            if name in ("vspace", "hspace"):
                _, i = take_arg(s, i)
                continue
            if name in DROP:
                continue
            if name in ("label", "index"):
                _, i = take_arg(s, i)
                continue
            if name in ("ref", "eqref", "autoref"):
                arg, i = take_arg(s, i)
                out.append(arg)
                continue
            if name in ("cite", "citep", "citet"):
                arg, i = take_arg(s, i)
                out.append("[" + arg + "]")
                continue
            if i < len(s) and s[i] == "{":
                arg, i = take_arg(s, i)
                out.append(parse(arg, math))
            continue

        if c == "$":
            j = s.find("$", i + 1)
            if j == -1:
                i += 1
                continue
            out.append(space_greek(parse(s[i + 1:j], True)))
            i = j + 1
        elif c == "{":
            j = match_brace(s, i)
            out.append(parse(s[i + 1:j], math))
            i = j + 1
        elif c == "}":
            i += 1
        elif c == "~":
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def clean(text: str) -> str:
    """Collapse a parsed fragment into one clean ASCII line."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("---", " - ").replace("--", "-")
    text = re.sub(r"(?<=[A-Za-z]) =(?=[\d-])", "=", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


GREEK_RE = re.compile(
    r"\b(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|rho|sigma|tau|phi|psi|"
    r"omega|Delta|Sigma|Theta)(?=[A-Za-z])")


def space_greek(text: str) -> str:
    """'deltatheta' -> 'delta theta'; applied only inside math fragments."""
    text = GREEK_RE.sub(r"\1 ", text)
    text = re.sub(r"\|\|\s+", "||", text)
    text = re.sub(r"\s*(<=|>=|!=)\s*", r" \1 ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\s*~\s*(?=[0-9])", " ~ ", text)
    text = re.sub(r"\s*->\s*", " -> ", text)
    return text


def inline(text: str) -> str:
    return clean(parse(text))


# --------------------------------------------------------------------------
# tables


def expand_multicolumn(cell: str) -> list[str]:
    m = re.match(r"\s*\\multicolumn\{(\d+)\}", cell)
    if not m:
        return [cell]
    span = int(m.group(1))
    spec = match_brace(cell, m.end())          # {c} / {l} alignment spec
    start = cell.index("{", spec + 1)          # the content group
    end = match_brace(cell, start)
    return [cell[start + 1:end]] + [""] * (span - 1)


def table_rows(chunk: str) -> list[list[str]]:
    chunk = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", chunk)
    chunk = re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", chunk)
    chunk = re.sub(r"\\(addlinespace|endhead|endfirsthead|endfoot|endlastfoot)(\[[^\]]*\])?", "", chunk)
    rows: list[list[str]] = []
    for raw in re.split(r"\\\\", chunk):
        if not raw.strip():
            continue
        cells: list[str] = []
        for cell in re.split(r"(?<!\\)&", raw):
            cells.extend(expand_multicolumn(cell))
        cells = [inline(c) for c in cells]
        if any(cells):
            rows.append(cells)
    return rows


def merge_header_rows(rows: list[list[str]]) -> list[str]:
    """Merge a sparse group row (e.g. CIFAR-100 / ImageNet-R) into the row below."""
    if len(rows) < 2:
        return rows[-1] if rows else []
    group = rows[0]
    sub = rows[1]
    n = max(len(group), len(sub))
    group += [""] * (n - len(group))
    sub += [""] * (n - len(sub))
    label = ""
    merged = []
    for g, s in zip(group, sub):
        if g:
            label = g
        merged.append(f"{label} {s}".strip())
    return merged


def render_grid(header: list[str], rows: list[list[str]]) -> str:
    ncols = max([len(header)] + [len(r) for r in rows])
    header = header + [""] * (ncols - len(header))
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(ncols)]
    total = sum(widths) + 3 * (ncols - 1)
    widest = max(len(c) for r in [header] + rows for c in r)

    if ncols <= 6 and total <= 120 and widest <= 48:
        def line(cells):
            return " | ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

        sep = "-+-".join("-" * w for w in widths)
        out = [line(header), sep]
        out += [line(r) for r in rows]
        return "\n".join(out)

    if ncols == 2:
        out = []
        for r in rows:
            line = f"{r[0]} - {r[1]}" if r[1] else r[0]
            out.append("\n".join(textwrap.wrap(
                line, width=WIDTH, subsequent_indent="   ", **WRAP_KW)))
        return "\n\n".join(out)

    # wide prose rows: one labelled block per row
    out = []
    for r in rows:
        if len(r[0]) <= 5 and len(r) > 1:
            head = f"{r[0]}. {r[1]}"
            rest = list(zip(header[2:], r[2:]))
        else:
            head = r[0]
            rest = list(zip(header[1:], r[1:]))
        block = textwrap.wrap(head, width=WIDTH, subsequent_indent="   ",
                              **WRAP_KW)
        for h, c in rest:
            if not c:
                continue
            label = f"   {h}: "
            block.extend(textwrap.wrap(
                label + c, width=WIDTH, subsequent_indent=" " * len(label),
                **WRAP_KW))
        out.append("\n".join(block))
    return "\n\n".join(out)


def render_table(body: str) -> str:
    if "\\midrule" in body:
        head_part, data_part = body.split("\\midrule", 1)
    else:
        head_part, data_part = "", body
    head_rows = table_rows(head_part)
    data_rows = table_rows(data_part)
    if not data_rows:
        return ""
    if len(head_rows) > 1:
        header = merge_header_rows(head_rows)
    elif head_rows:
        header = head_rows[-1]
    else:
        header = [""] * len(data_rows[0])
    # drop a header that ended up empty in every cell
    if not any(header):
        header = [""] * len(data_rows[0])
    return render_grid(header, data_rows)


# --------------------------------------------------------------------------
# block environments


def inline_items(content: str) -> list[str]:
    parts = re.split(r"\\item\b(\[[^\]]*\])?", content)
    # re.split with a capturing group returns [pre, group1, item1, group2, item2, ...]
    items = [p for i, p in enumerate(parts) if i % 2 == 0][1:]
    return [p for p in items if p.strip()]


def render_list(content: str, ordered: bool, style: str) -> str:
    out = []
    for idx, item in enumerate(inline_items(content), 1):
        text = inline(item.replace("\\item", ""))
        if ordered:
            marker = f"({ROMAN[idx]}) " if style == "roman" else f"{idx}. "
        else:
            marker = "- "
        wrapped = textwrap.wrap(text, width=WIDTH, initial_indent=marker,
                                subsequent_indent=" " * len(marker), **WRAP_KW)
        out.append("\n".join(wrapped) if wrapped else marker.rstrip())
    return "\n".join(out)


def render_quote(content: str, prefix: str = "> ") -> str:
    text = inline(content)
    return "\n".join(textwrap.wrap(text, width=WIDTH - len(prefix),
                                   initial_indent=prefix,
                                   subsequent_indent=prefix, **WRAP_KW))


class Doc:
    def __init__(self) -> None:
        self.store: list[str] = []
        self.sec = 0

    def keep(self, text: str) -> str:
        self.store.append(text.rstrip())
        return f"\n\n@@B{len(self.store) - 1}@@\n\n"

    def extract_tables(self, src: str) -> str:
        pattern = re.compile(r"\\begin\{(tabular|longtable)\}")
        while True:
            m = pattern.search(src)
            if not m:
                return src
            env = m.group(1)
            j = m.end()
            if j < len(src) and src[j] == "[":
                j = src.index("]", j) + 1
            k = match_brace(src, j)
            end = src.index("\\end{%s}" % env, k)
            self.keep(render_table(src[k + 1:end]))
            src = src[:m.start()] + f"\n\n@@B{len(self.store) - 1}@@\n\n" + \
                src[end + len("\\end{%s}" % env):]

    def sub_env(self, src: str, name: str, fn) -> str:
        pattern = re.compile(r"\\begin\{%s\}(\s*\[[^\]]*\])?(.*?)\\end\{%s\}"
                             % (name, name), re.S)
        while True:
            m = pattern.search(src)
            if not m:
                return src
            src = src[:m.start()] + self.keep(fn(m.group(2), m.group(1) or "")) + \
                src[m.end():]

    def sub_heading(self, src: str, name: str, render) -> str:
        pattern = re.compile(r"\\%s(\*?)\{" % name)
        pos = 0
        out = []
        while True:
            m = pattern.search(src, pos)
            if not m:
                out.append(src[pos:])
                break
            out.append(src[pos:m.start()])
            j = match_brace(src, m.end() - 1)
            title = clean(parse(src[m.end():j]))
            out.append(self.keep(render(title, m.group(1) == "*")))
            pos = j + 1
        return "".join(out)


def build(src_root: Path, files: list[str] | None = None,
          numbered: bool = True) -> str:
    main = (src_root / "main.tex").read_text(encoding="utf-8")
    assert "NEUCOM-D-26-24287" in main, "manuscript id drifted from main.tex"
    body = "\n".join(
        (src_root / name).read_text(encoding="utf-8")
        for name in (files or LETTER_FILES))

    doc = Doc()
    body = doc.extract_tables(body)
    body = doc.sub_env(body, "rquote", lambda c, o: render_quote(c))
    body = doc.sub_env(body, "quote", lambda c, o: render_quote(c, "  "))
    body = doc.sub_env(body, "itemize", lambda c, o: render_list(c, False, ""))
    body = doc.sub_env(
        body, "enumerate",
        lambda c, o: render_list(c, True, "roman" if "roman" in o else "arabic"))
    body = re.sub(r"\\begin\{center\}(.*?)\\end\{center\}", r"\n\1\n", body, flags=re.S)

    def sec_render(title: str, starred: bool) -> str:
        if starred or not numbered:
            head = title.upper()
        else:
            doc.sec += 1
            head = f"{doc.sec}. {title.upper()}"
        rule = "=" * 78
        return f"{rule}\n{head}\n{rule}"

    def sub_render(title: str, starred: bool) -> str:
        return title

    body = doc.sub_heading(body, "section", sec_render)
    body = doc.sub_heading(body, "subsection", sub_render)
    body = doc.sub_heading(body, "subsubsection", sub_render)

    body = body.replace("\\resp", "\n\n@@RESP@@ ")
    body = body.replace("\\change", "\n\n@@CHANGE@@ ")
    body = re.sub(r"\\begin\{document\}|\\end\{document\}", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)

    chunks = re.split(r"\n\s*\n", body)
    paras = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.fullmatch(r"@@B\d+@@", chunk):
            paras.append(chunk)
            continue
        if chunk.startswith("@@RESP@@"):
            text = "Response. " + inline(chunk[len("@@RESP@@"):])
        elif chunk.startswith("@@CHANGE@@"):
            text = "Change in the manuscript. " + inline(chunk[len("@@CHANGE@@"):])
        else:
            text = inline(chunk)
        text = " ".join(text.split())
        if not text:
            continue
        paras.append(textwrap.fill(text, width=WIDTH, **WRAP_KW))

    for idx, text in enumerate(doc.store):
        paras = [p.replace(f"@@B{idx}@@", text) for p in paras]

    header = (
        "RESPONSE TO THE REVIEWERS\n\n"
        "Manuscript NEUCOM-D-26-24287\n\n"
        "FOLoRA: Fisher-Orthogonalized Low-Rank Adaptation for Continual Learning\n\n"
        "Neurocomputing | Jiang Tao (corresponding author) | 8 November 2026\n\n"
        + "-" * 78
    )
    out = header + "\n\n" + "\n\n".join(paras) + "\n"
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.replace("Jiang Tao Corresponding author 8 November 2026",
                      "Jiang Tao\nCorresponding author\n8 November 2026")
    return out


# One text per reviewer, each self-contained: same front matter, no section
# numbering (a lone "3. Response to Reviewer 5" reads as a missing page 2).
SPLIT = [
    ("response-editor-and-summary.txt", ["01_summary.tex", "07_files.tex"]),
    ("response-reviewer-1.txt", ["02_reviewer1.tex"]),
    ("response-reviewer-3.txt", ["03_reviewer3.tex"]),
    ("response-reviewer-5.txt", ["04_reviewer5.tex"]),
    ("response-reviewer-7.txt", ["05_reviewer7.tex"]),
    ("response-reviewer-8.txt", ["06_reviewer8.tex"]),
]


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    words = len(text.split())
    print(f"wrote {path.name}  ({len(text):,} bytes, {words:,} words, "
          f"{text.count(chr(10)) + 1:,} lines)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--out", required=True, help="directory for the .txt file")
    ap.add_argument("--name", default="response-to-reviewers-paste-full.txt")
    ap.add_argument("--split", action="store_true",
                    help="write one file per reviewer instead of one combined file")
    args = ap.parse_args()

    src_root = Path(args.repo).resolve() / "paper" / "response-letter"
    main_tex = (src_root / "main.tex").read_text(encoding="utf-8")
    assert "NEUCOM-D-26-24287" in main_tex, "manuscript id drifted from main.tex"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        for name, files in SPLIT:
            write(out_dir / name,
                  build(src_root, files, numbered=False))
    else:
        write(out_dir / args.name, build(src_root))


if __name__ == "__main__":
    main()
