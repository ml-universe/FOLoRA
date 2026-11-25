"""Generate the two author-side submission documents from the repository.

The first submission uploaded ``author-biography.docx`` and ``Declaration of
competing interests.docx`` as hand-made Word files; those originals are gone.
The text of both still exists: the biography is the one printed on page 132 of
the composed R1 submission (kept in ``paper/author-biography.txt``, unchanged
apart from the en dash in ``vision-language``), and the declaration is a section
of ``paper/main.tex``. So both are generated from the repository rather than
re-typed, and the wording cannot drift from the manuscript.

The photograph cannot be generated; it is extracted from the composed R1 PDF
(page 133, 480x720 JPEG, 16,936 bytes) and passed with ``--photo``.

Usage:
    py -3.11 scripts/make_author_docs.py --out <dir> --photo <author-photo.jpg>

Outputs, in ``--out``:
    author-biography.docx
    Declaration of competing interests.docx
    author-photo.jpg            (copied from --photo, if given)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

# Runnable as `py -3.11 scripts/make_author_docs.py` as well as with -m: the
# script directory is on sys.path, not the repository root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_submission_docs import _default_template, build  # noqa: E402

PAPER = ROOT / "paper"
BIO_TXT = PAPER / "author-biography.txt"


def read_bio() -> list[str]:
    paragraphs = [p.strip() for p in BIO_TXT.read_text(encoding="utf-8").split("\n\n")]
    return [p for p in paragraphs if p]


# Elsevier's standard declaration form, as printed on page 140 of the composed R1
# submission: two statements, the first box checked. The manuscript carries the
# same declaration as a section; this file is the form the system expects.
_DECLARATION = [
    "☒  The authors declare that they have no known competing financial interests or "
    "personal relationships that could have appeared to influence the work reported in this "
    "paper.",
    "☐  The authors declare the following financial interests/personal relationships "
    "which may be considered as potential competing interests:",
    "",
    "",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--photo", default=None, help="author-photo.jpg to copy in")
    ap.add_argument("--template", default=None,
                    help="docx to copy styling from (default: generated plain one)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.template and not Path(args.template).exists():
        raise SystemExit(f"template not found: {args.template}")
    if args.template:
        tpl = Path(args.template)
    else:
        tpl = _default_template(Path(tempfile.mkdtemp(prefix="author-docx-tpl-"))
                                / "template.docx")

    bio = read_bio()
    words = len(" ".join(bio).split())
    print(f"biography: {words} words in {len(bio)} paragraphs")
    build(tpl, "Author biography", bio, out / "author-biography.docx")

    decl = _DECLARATION
    build(tpl, "Declaration of interests", decl,
          out / "Declaration of competing interests.docx")

    if args.photo:
        src = Path(args.photo)
        if not src.exists():
            raise SystemExit(f"photo not found: {src}")
        shutil.copyfile(src, out / "author-photo.jpg")
        print(f"photo: copied {src.name} ({src.stat().st_size:,} bytes)")

    print(f"\nwrote author files to {out}")


if __name__ == "__main__":
    main()
