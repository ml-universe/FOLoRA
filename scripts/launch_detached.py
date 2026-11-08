"""把一条训练命令**真正**放到后台跑，脱离当前控制台/进程组。

为什么需要它
------------
2026-11-02 试过的两种「后台」都不成立：

* `cmd /c start /min cmd /c ...` —— 新开的控制台窗口属于本会话，Bash 工具调用一结束
  窗口就被关掉，子进程收到 CTRL_CLOSE_EVENT，fortran 运行时直接报
  `forrtl: error (200): program aborting due to window-CLOSE event`，训练在 task 1
  中途死掉（日志有据，见 logs/unweighted_topup_20261102.log 的第一次尝试）。
* `start /b` —— 不新建控制台，Ctrl-C/进程组信号直接打到子进程上，更糟。

正确做法是 DETACHED_PROCESS（0x8）：子进程没有控制台，不接收本会话的 CTRL_CLOSE /
CTRL_C，父进程退出也不连带。stdio 全部重定向到日志文件，因此不需要控制台。

用法
----
    py -3.11 scripts/launch_detached.py --log logs/xxx.log -- \
        -m scripts.run_lambda_grid --batch unweighted

`--` 之后是本解释器要执行的参数（也会用同一个 sys.executable 执行）。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Windows 常量；非 Windows 上不生效，脚本只在本机（Windows）用。
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="stdout/stderr 重定向到这个文件（追加）")
    p.add_argument("--cwd", default=".", help="子进程工作目录")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    args = p.parse_args()

    rest = args.rest
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        p.error("`--` 之后要给出要执行的参数，例如 `-- -m scripts.run_lambda_grid ...`")

    log = Path(args.log).resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    # 追加：续跑/重试都写同一个文件，历史不丢
    fh = open(log, "ab")
    env = dict(os.environ)
    # 子进程没有控制台，Python 会按 locale（cp936）编码 stdout；日志里就会一半 GBK
    # 一半 UTF-8，中文全成乱码。钉成 UTF-8，`tail`/`grep` 才读得懂。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", *rest],
        cwd=str(Path(args.cwd).resolve()),
        stdin=subprocess.DEVNULL,
        stdout=fh,
        stderr=subprocess.STDOUT,
        close_fds=True,
        env=env,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )
    fh.close()          # 父进程这份句柄可以关，子进程仍持有自己的
    print(f"launched pid={proc.pid}  log={log}")


if __name__ == "__main__":
    main()
