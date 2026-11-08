"""在训练期间压住系统的**空闲**休眠，直到指定 pid 退出。

为什么需要它
------------
2026-11-01/11-02 的 Kernel-Power 事件显示机器每晚 22:29 左右进入睡眠（ID 187 →
42），唤醒要到次日 11:45，一夜丢十几个小时。电源空闲超时早已设为「从不」，所以
这不是超时触发的；但训练进程本身不持有任何电源请求，系统一旦被判定空闲就照睡。

`SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)` 是**非管理员**也能持有的
电源请求，作用范围仅限本进程存活期间，退出即释放，不改任何系统设置。

**它挡不住合盖休眠**：合盖动作是系统级的强制转换，需要用管理员的
`powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0` 才挡得住。
所以本脚本只解决「夜间空闲自动睡眠」，不解决「合盖」。

用法：
    py -3.11 scripts/launch_detached.py --log logs/keep_awake.log --cwd . -- \
        -m scripts.keep_awake --pid 20652
"""

import argparse
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102


def pid_alive(pid: int) -> bool:
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(SYNCHRONIZE, False, pid)
    if not h:
        return False
    try:
        return k32.WaitForSingleObject(h, 0) == WAIT_TIMEOUT
    finally:
        k32.CloseHandle(h)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True, help="该进程退出后释放电源请求")
    p.add_argument("--poll", type=int, default=30)
    p.add_argument("--max-hours", type=float, default=30.0)
    args = p.parse_args()

    k32 = ctypes.windll.kernel32
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    started = time.time()

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | 持有电源请求，直到 pid {args.pid} 退出",
          flush=True)
    while True:
        if k32.SetThreadExecutionState(flags) == 0:
            # 返回 0 表示调用失败；不致命，但要留痕，否则会误以为在生效
            print(f"{time.strftime('%H:%M:%S')} | SetThreadExecutionState 返回 0（失败）",
                  flush=True)
        if not pid_alive(args.pid):
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | pid {args.pid} 已退出，释放请求",
                  flush=True)
            break
        if (time.time() - started) / 3600 > args.max_hours:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | 到达 max-hours，主动释放",
                  flush=True)
            break
        time.sleep(args.poll)

    k32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
