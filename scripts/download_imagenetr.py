"""从 ModelScope 下载并解压 ImageNet-R（OpenDataLab/ImageNet-R）。

ImageNet-R：200 类、~3 万张域偏移图像（卡通/涂鸦等），持续学习常用基准。
数据在 raw/imagenet-r.tar（约 2.19GB），解压后是 200 个类别文件夹（每类若干 jpg）。

用法：python -m scripts.download_imagenetr
"""

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
    with tarfile.open(tar_path, "r") as tf:
        # 只解压文件（跳过路径穿越等），安全提取
        for member in tf:
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                continue
            tf.extract(member, out_dir, filter="data")
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
