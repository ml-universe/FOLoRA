"""从 ModelScope 下载 CIFAR-100（parquet）到 data/cifar100/，幂等。

cs.toronto.edu 在国内极慢（~8kB/s），改用 modelscope 镜像（~8MB/s）。
下载的 parquet 直接供 src/peft_cl/data/cifar_parquet.py 读取。

用法：python -m scripts.download_cifar
"""

import urllib.request
from pathlib import Path

# (本地文件名, ModelScope 上的 FilePath)
FILES = {
    "train.parquet": "cifar100%2Ftrain-00000-of-00001.parquet",
    "test.parquet": "cifar100%2Ftest-00000-of-00001.parquet",
}
BASE_URL = ("https://modelscope.cn/api/v1/datasets/cutedataset/cifar100/repo"
            "?Source=SDK&Revision=master&FilePath=")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def main():
    out_dir = Path("data") / "cifar100"
    for local, fp in FILES.items():
        dest = out_dir / local
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {dest} already exists")
            continue
        download(BASE_URL + fp, dest)
    print("done.")


if __name__ == "__main__":
    main()
