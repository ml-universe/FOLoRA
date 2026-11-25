"""Generate the submission Word files (abstract + highlights) from paper/main.tex.

The editorial system wants .docx for the abstract and the highlights, but the
manuscript is the single source of truth: the revision changed both (change C1),
and a hand-edited .docx drifts silently. This script reads the two blocks out of
``paper/main.tex``, flattens the small amount of LaTeX they contain, and writes
two documents that inherit their styling from the documents submitted in the
first round.

The first-round .docx files are no longer on disk (only their rendered pages 3--4
of the composed submission PDF survive). When no template is passed, a plain one
is generated instead: these two files are reviewed for their text, not their look,
and refusing to build over a missing style carrier would block the submission.
Pass ``--abstract-template`` / ``--highlights-template`` to reuse the originals if
they are recovered.

Usage:
    py -3.11 scripts/make_submission_docs.py --out <dir>

Outputs, in ``--out``:
    abstract.docx
    highlights.docx
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

import docx

ROOT = Path(__file__).resolve().parent.parent
MAIN_TEX = ROOT / "paper" / "main.tex"


def _flatten(tex: str) -> str:
    """Flatten the LaTeX that can appear in an abstract or a highlight."""
    s = tex
    s = re.sub(r"%.*", "", s)                       # comments
    s = s.replace(r"$\sim$", "~")
    s = re.sub(r"\\sim", "~", s)
    s = re.sub(r"\\times", "\u00d7", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)            # inline math -> contents
    s = re.sub(r"\\(?:textbf|textit|emph|text)\{([^{}]*)\}", r"\1", s)
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("--", "\u2013")
    s = s.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)            # any remaining command
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_blocks() -> tuple[str, list[str]]:
    tex = MAIN_TEX.read_text(encoding="utf-8")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        raise SystemExit("abstract block not found in main.tex")
    abstract = _flatten(m.group(1))

    m = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", tex, re.S)
    if not m:
        raise SystemExit("highlights block not found in main.tex")
    items = [_flatten(x) for x in re.findall(r"\\item\s+(.*?)(?=\\item|$)", m.group(1), re.S)]
    items = [i for i in items if i]
    if not items:
        raise SystemExit("highlights block parsed to zero items")
    return abstract, items


def _default_template(path: Path) -> Path:
    """Write a plain Letter / Times New Roman 12 pt template.

    Used only when the first-round .docx files cannot be supplied. Their
    composed form in the R1 submission PDF is a Letter page of Times New Roman
    body text; this reproduces that shape without pretending to copy anything
    that no longer exists.
    """
    d = docx.Document()
    normal = d.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = docx.shared.Pt(12)
    for section in d.sections:
        section.page_width = docx.shared.Inches(8.5)
        section.page_height = docx.shared.Inches(11)
        section.top_margin = section.bottom_margin = docx.shared.Inches(1)
        section.left_margin = section.right_margin = docx.shared.Inches(1)
    d.save(str(path))
    return path


def _blank_document(body: docx.document.Document) -> None:
    """Clear all body paragraphs, keeping styles and section setup."""
    body_el = body.element.body
    for child in list(body_el):
        if child.tag.endswith("}p"):
            body_el.remove(child)


def build(template: Path, heading: str, paragraphs: list[str], out: Path,
          bullet: bool = False) -> None:
    shutil.copyfile(template, out)
    d = docx.Document(str(out))
    _blank_document(d)

    h = d.add_paragraph()
    run = h.add_run(heading)
    run.bold = True
    run.font.size = docx.shared.Pt(14)

    for text in paragraphs:
        p = d.add_paragraph(style="List Bullet" if bullet else None)
        p.add_run(text)

    d.save(str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--abstract-template", default=None,
                    help="docx to copy styling from (default: search the desktop pack)")
    ap.add_argument("--highlights-template", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    abstract, items = read_blocks()
    print(f"abstract: {len(abstract.split())} words")
    for i, it in enumerate(items, 1):
        print(f"  highlight {i} ({len(it)} chars): {it}")

    a_tpl = Path(args.abstract_template) if args.abstract_template else None
    h_tpl = Path(args.highlights_template) if args.highlights_template else None
    for name, tpl in (("abstract", a_tpl), ("highlights", h_tpl)):
        if tpl is not None and not tpl.exists():
            raise SystemExit(f"{name} template not found: {tpl}")
    tpl_dir = Path(tempfile.mkdtemp(prefix="submission-docx-tpl-"))
    if a_tpl is None:
        a_tpl = _default_template(tpl_dir / "abstract-template.docx")
        print("note: no abstract template given; using a generated "
              "Letter / Times New Roman 12 pt one")
    if h_tpl is None:
        h_tpl = _default_template(tpl_dir / "highlights-template.docx")
        print("note: no highlights template given; using a generated "
              "Letter / Times New Roman 12 pt one")

    build(a_tpl, "Abstract", [abstract], out / "abstract.docx")
    build(h_tpl, "Highlights", items, out / "highlights.docx", bullet=True)
    print(f"\nwrote 2 files to {out}")


if __name__ == "__main__":
    main()
