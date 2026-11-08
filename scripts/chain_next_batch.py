"""等一个跑着的 pid 退出 → 跑闸门校验 → 校验通过才启动下一批训练。

为什么要有它
------------
2026-11-02 大修的补实验是**串行**的：E1（等权补种子，13 run）先占满 GPU，E2（真实
遗忘轨迹，9 run × 约 2 h）本来要等 E1 结束才能开。等人回来点一下是浪费——夜里 GPU
会空转 10 小时以上。这个脚本让「等人」变成「等进程」，但**不是无条件下一条**：
E2 的正确性依赖于它和主表 NCM 同口径，所以中间插一道 `validate_ncm_trace.py`，
不过关就**不启动**，只把结果写进日志（宁可空转，也不要花 20 小时跑一套不可比的曲线）。

用法（本身要用 launch_detached 起，否则它自己也会被会话关闭杀掉）：

    py -3.11 scripts/launch_detached.py --log logs/chain_E1_E2.log --cwd . -- \
        -m scripts.chain_next_batch --wait-pid 20652 \
        --validate "--method folora_v2 --tag v2f_l3_k64 --seed 0" \
        --batch ncm_trace

2026-11-03 起 `--batch` 接受多批（空格分隔，按序串行）：

    ... --wait-pid 29464 --batch olora_orth_topup repro_floor

多批共用**同一个守护进程**，所以 keep_awake 只要绑这一个 pid 就能压住整条链的空闲
休眠；每批各起一个 chain 脚本会在切换处留一个「无电源请求」的窗口。任一批非 0 退出
即中止后续批次。
"""

import argparse
import ctypes
import shlex
import subprocess
import sys
import time

SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102


def pid_alive(pid: int) -> bool:
    """Windows 上用 OpenProcess+WaitForSingleObject 判活，不依赖 psutil。

    比 `tasklist | grep` 稳：不会因为同名进程（同一台机器上可能还有别的 python）误判，
    也能正确处理「进程已退出但任务列表缓存还没刷新」。
    """
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
    p.add_argument("--wait-pid", type=int, required=True)
    p.add_argument("--poll", type=int, default=60, help="轮询间隔（秒）")
    p.add_argument("--validate", default="", help="validate_ncm_trace.py 的参数串，空=跳过校验")
    # 2026-11-03：改成可接多批，用 space-separated 顺序串起来（`--batch E6 E4`）。
    # 目的只有一个：让**一条守护进程**贯穿多批，keep_awake 只要绑这一个 pid 就能压住
    # 整条链的空闲休眠。若每批各起一个 chain 脚本，前一个退出 → 它绑的 keep_awake
    # 释放 → 下一个脚本在「没有电源请求」的窗口里启动，半夜正好可能睡过去。
    p.add_argument("--batch", nargs="+", default=[],
                   help="校验通过后依次要跑的 run_lambda_grid 批次（空=不跑）")
    # 2026-11-07：加一个「不属于任何批次」的收尾钩子。A10（峰值显存实测）跑的是
    # `scripts.measure_peak_mem`，不是 run_lambda_grid 的批，没法塞进 --batch；
    # 而它又必须等 GPU 空出来才能测（否则读到别的 run 的占用）。
    p.add_argument("--cmd", action="append", default=[], metavar="CMD",
                   help="**所有批次跑完之后**要执行的命令（可重复，按序串行）。"
                        "按 shell 规则分词，例如 `--cmd \"-m scripts.measure_peak_mem\"`。"
                        "放在批次之后而不是之前，是为了不让它抢 GPU。")
    args = p.parse_args()

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)

    log(f"等待 pid {args.wait_pid} 退出（每 {args.poll}s 查一次）")
    while pid_alive(args.wait_pid):
        time.sleep(args.poll)
    log(f"pid {args.wait_pid} 已退出，开始下一步")

    if args.validate:
        cmd = [sys.executable, "-u", "scripts/validate_ncm_trace.py", *args.validate.split()]
        log(f"校验：{' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            log(f"校验未通过（退出码 {rc}）——**不启动** {'/'.join(args.batch) or '任何批次'}。"
                f"先人工看 ncm_trace 与 eval_ncm 的口径差异。")
            # 非 0 退出，便于事后一眼看出「这条链是主动停的，不是崩的」
            sys.exit(3)
        log("校验通过")

    # 多批串行：前一批非 0 退出时**停止后续批次**（不是继续往下跑）。理由是一批
    # 失败通常意味着环境问题（OOM、数据路径、依赖），后面的批会以同样方式失败，
    # 继续跑只会在日志里刷同样的栈。停在这里由人看一眼更便宜。
    for batch in args.batch:
        cmd = [sys.executable, "-u", "-m", "scripts.run_lambda_grid", "--batch", batch]
        log(f"启动批次：{' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        log(f"批次 {batch} 结束，退出码 {rc}")
        if rc != 0:
            log(f"批次 {batch} 非 0 退出 —— 中止链，后续批次不启动")
            sys.exit(rc)

    # 收尾钩子：任意命令（A10 峰值显存实测走这里）。失败同样中止并留痕——
    # 静默失败比不跑更糟：第二天会以为「跑过了」。
    for cmdline in args.cmd:
        cmd = [sys.executable, "-u", *shlex.split(cmdline)]
        log(f"执行收尾命令：{' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        log(f"收尾命令结束，退出码 {rc}")
        if rc != 0:
            log("收尾命令非 0 退出 —— 中止链")
            sys.exit(rc)

    log("链结束")


if __name__ == "__main__":
    main()
