"""Package the manuscript's figures as standalone files for the journal.

The decision letter asks for "source files ... as well as separate figure files;
these will be used for the eventual typesetting of the paper".  Fig. 1 is TikZ
inside the manuscript, so it has no file of its own: this script compiles it in
isolation, crops the result, and puts it in an archive next to the two Appendix-C
figures, which already exist as vector PDFs.

Usage:
    py -3.11 scripts/make_figures_zip.py --out <dir>

Writes <dir>/figures.zip.  Requires latexmk and pdfcrop on PATH (TeX Live).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.stdout.reconfigure(errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "paper" / "figures"

# Fig. 1 is a `figure` float, so it cannot go into `standalone` directly.
WRAPPER = r"""\documentclass[11pt]{article}
\usepackage[paperwidth=190mm,paperheight=170mm,margin=6mm]{geometry}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
\usepackage{amsmath,amssymb}
\pagestyle{empty}
\begin{document}
\input{framework}
\end{document}
"""

README = """FOLoRA -- separate figure files (NEUCOM-D-26-24287, major revision, 2026-11-08)

Fig. 1    framework              fig1_framework.pdf (vector, cropped) + fig1_framework.tex (TikZ source)
Fig. C.1  candidate-set curves   figC1_forgetting_cifar100.pdf (+ .png raster copy)
Fig. C.2  candidate-set curves   figC2_forgetting_imagenetr.pdf (+ .png raster copy)

These are the same assets that are compiled into main.pdf; no figure was edited
after the manuscript build.  The vector PDFs are the intended production format.
The PNG copies are provided only for convenience and are not the source of any
number in the paper.
"""


def run(cmd: list[str], cwd: pathlib.Path) -> None:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{res.stdout[-3000:]}\n{res.stderr[-2000:]}")


def build_fig1(tmp: pathlib.Path) -> pathlib.Path:
    shutil.copyfile(FIGDIR / "framework.tex", tmp / "framework.tex")
    (tmp / "fig1.tex").write_text(WRAPPER, encoding="utf-8")
    run(["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "fig1.tex"], tmp)
    run(["pdfcrop", "--margins", "4", "fig1.pdf", "fig1_framework.pdf"], tmp)
    return tmp / "fig1_framework.pdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        items: list[tuple[pathlib.Path, str]] = [(build_fig1(tmp), "fig1_framework.pdf")]
        items.append((FIGDIR / "framework.tex", "fig1_framework.tex"))
        for src, arc in (
            ("forgetting_curve_cifar100.pdf", "figC1_forgetting_cifar100.pdf"),
            ("forgetting_curve_cifar100.png", "figC1_forgetting_cifar100.png"),
            ("forgetting_curve_imagenetr.pdf", "figC2_forgetting_imagenetr.pdf"),
            ("forgetting_curve_imagenetr.png", "figC2_forgetting_imagenetr.png"),
        ):
            items.append((FIGDIR / src, arc))

        readme = tmp / "00_README.txt"
        readme.write_text(README, encoding="utf-8")
        items.insert(0, (readme, "00_README.txt"))

        dest = out / "figures.zip"
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for src, arc in items:
                zf.write(src, arc)
        print(f"  figures.zip: {len(items)} files, {dest.stat().st_size/1024:.0f} KiB")


if __name__ == "__main__":
    main()
