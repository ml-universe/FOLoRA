"""生成 paper/main-marked-up.pdf（latexdiff：上一轮送审稿 → 本次修改稿）。

为什么要有脚本，而不是在临时目录里手敲
--------------------------------------
标注稿是投稿清单里的第 2 项，且必须在**最终稿**上重做一次：数字一改，旧标注稿即作废。
手敲流程有三个隐形前提（旧版取自 `HEAD:paper` 而不是工作区、`--flatten` 后仍需
`refs.bib`/`main.bbl`/`figures/` 才能编译、编译要在 `new/` 里做），任何一条记错，
结果要么是把「本轮改动」标成没改，要么是编译悄悄退回无图无参考文献的版本。
所以做成脚本：`--old-ref` 默认 `HEAD`，即上一轮实际送出去的那一版。

用法
----
    py -3.11 -m scripts.build_marked_up
    py -3.11 -m scripts.build_marked_up --old-ref <sha> --out <path> --keep

`--keep` 保留工作目录，便于肉眼核对 marked.tex。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"

# 只带走能编译的东西；aux 类产物如果带进去，latexmk 会以为增量状态有效而跳过重跑。
# 两点别踩：不能按 `_*` 过滤（sections/_figs_candidate_curves.tex 是正文 \input 的图源），
# 也不能按 `*.pdf` 过滤（figures/ 下的图就是 pdf，过滤了图全变成 draft 框）。
IGNORE = shutil.ignore_patterns(
    "*.aux", "*.log", "*.fdb_latexmk", "*.fls", "*.out", "*.spl", "*.blg",
    "cover-letter*", "main-marked-up*",
)


def sh(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-ref", default="HEAD",
                    help="旧版 git 引用；默认 HEAD（上一轮送审的那一版）")
    ap.add_argument("--out", default=str(PAPER / "main-marked-up.pdf"))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="latexdiff-"))
    old, new = work / "old", work / "new"

    # 1) 旧版：直接取 git 里的 paper/ 子树，不能拿工作区当旧版（那样结果里一处改动都没有）
    old.mkdir()
    tar = work / "paper-old.tar"
    sh(["git", "archive", "-o", str(tar), f"{args.old_ref}:paper"], cwd=REPO)
    sh(["tar", "-xf", str(tar), "-C", str(old)])

    # 2) 新版：工作区副本（排除编译产物）
    shutil.copytree(PAPER, new, ignore=IGNORE)
    (new / "main.pdf").unlink(missing_ok=True)  # 上一轮的成品，别混进来

    # 3) latexdiff --flatten：\input 全部内联。
    #    两个必须整段取新版的环境（都写进了回复信「文件清单」的说明里）：
    #    - thebibliography：latexdiff 会把 url 宏保护 \csname url\endcsname 拆进
    #      \DIFadd{}，报 "Missing \endcsname inserted" 直接编译失败；
    #    - table/tabular：\DIFaddendFL 落在 \bottomrule 前会破坏 booktabs 的
    #      \noalign，报 "Misplaced \noalign"。
    #    两者的增删本来就在回复信里逐条说明，标注稿按新版呈现即可。
    #    第三个坑更隐蔽：latexdiff 默认只给 \cite 家族套 \mbox，\ref 不套。
    #    第一遍（引用未解析）能过，第二遍 hyperref 把 \ref 展开成
    #    \hyperlink 的 \begingroup...\endgroup，被 ulem 的 \sout 吞掉，
    #    报 "\UL@stop ... Extra }"。所以 \ref/\eqref 也加进 mbox 名单。
    marked = new / "main-marked-up.tex"
    with marked.open("w", encoding="utf-8") as fh:
        r = subprocess.run(
            ["latexdiff", "--flatten",
             "--config=PICTUREENV=(?:picture|table|tabular|thebibliography)",
             "--append-mboxsafecmd=ref,eqref,autoref",
             str(old / "main.tex"), str(new / "main.tex")],
            cwd=work, text=True, stdout=fh, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        return 1

    tex = marked.read_text(encoding="utf-8")
    n_add = len(re.findall(r"\\DIFaddbegin", tex))
    n_del = len(re.findall(r"\\DIFdelbegin", tex))
    print(f"latexdiff: {n_add} 处新增、{n_del} 处删除（{len(tex):,} 字符）")
    if n_add == 0 and n_del == 0:
        print("两版完全一致 —— 旧引用选错了？", file=sys.stderr)
        return 1

    # 4) 在 new/ 里编译：figures/、refs.bib、main.bbl、elsarticle.cls 都相对 main.tex 解析
    r = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-jobname=main-marked-up", "main-marked-up.tex"],
        cwd=new, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(r.stdout[-4000:], file=sys.stderr)
        return 1

    pdf = new / "main-marked-up.pdf"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, out)

    log = (new / "main-marked-up.log").read_text(encoding="utf-8", errors="replace")
    bad = [ln for ln in log.splitlines()
           if re.search(r"(Undefined control sequence|LaTeX Error|! )", ln)]
    pages = re.search(r"Output written on main-marked-up\.pdf \((\d+) pages", log)
    print(f"→ {out}  {pages.group(1) + ' 页' if pages else '(页数未解析到)'}")
    if bad:
        print("编译日志里有错误行：", file=sys.stderr)
        print("\n".join(bad[:10]), file=sys.stderr)
    if args.keep:
        print(f"工作目录保留：{work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
