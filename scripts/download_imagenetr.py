"""从 ModelScope 下载并解压 ImageNet-R（OpenDataLab/ImageNet-R）。

ImageNet-R：200 类、~3 万张域偏移图像（卡通/涂鸦等），持续学习常用基准。
数据在 raw/imagenet-r.tar（约 2.19GB），解压后是 200 个类别文件夹（每类若干 jpg）。

用法：python -m scripts.download_imagenetr
"""

import os
import tarfile
import urllib.request
from pathlib import Path

URL = ("https://modelscope.cn/api/v1/datasets/OpenDataLab/ImageNet-R/repo"
       "?Source=SDK&Revision=master&FilePath=raw%2Fimagenet-r.tar")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading -> {dest} ({url[:80]}...)")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r  {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.replace(dest)


def extract(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting -> {out_dir}")
    out_root = out_dir.resolve()
    warned_no_filter = False
    with tarfile.open(tar_path, "r") as tf:
        # 只解压文件（跳过路径穿越等），安全提取
        for member in tf:
            target = (out_dir / member.name).resolve()
            # 路径穿越检查用 commonpath，**不是** str.startswith：
            # startswith 会把同前缀的兄弟目录误判为在内（out_dir=/data/imagenetr 时
            # /data/imagenetr-evil/... 也以同一前缀开头）。commonpath 在不同盘符上
            # 抛 ValueError，此时必然在 out_dir 之外。
            try:
                inside = os.path.commonpath([str(target), str(out_root)]) == str(out_root)
            except ValueError:
                inside = False
            if not inside:
                continue
            try:
                # filter="data" 由 Python 3.12 引入并回移到 3.11.4+，会在解压阶段
                # 再拦一道路径穿越/危险成员。解释器更旧时该关键字不存在，会抛
                # TypeError —— 此时不能让已下完的 2.19 GB 白费，退回不带 filter 的调用。
                tf.extract(member, out_dir, filter="data")
            except TypeError:
                if not warned_no_filter:
                    print("  [警告] 当前解释器不支持 filter='data'（需 Python >= 3.11.4）；"
                          "退回不带 filter 的解压，路径穿越防护仅剩上面的 commonpath 检查。")
                    warned_no_filter = True
                tf.extract(member, out_dir)
    print(f"done. 类别文件夹数：{len([d for d in out_dir.iterdir() if d.is_dir()])}")


def main():
    out_dir = Path("data") / "imagenetr"
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"[skip] already extracted: {out_dir}")
        return
    tar_path = Path("data") / "raw" / "imagenet-r.tar"
    if not tar_path.exists() or tar_path.stat().st_size < 2_000_000_000:
        download(URL, tar_path)
    extract(tar_path, out_dir)
    marker.touch()


if __name__ == "__main__":
    main()
