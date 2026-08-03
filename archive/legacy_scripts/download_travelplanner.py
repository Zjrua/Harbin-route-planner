"""下载 TravelPlanner 数据集（OSU NLP, NeurIPS 2024）.

通过 hf-mirror.com 镜像下载（huggingface.co 主站国内被墙）。
CC-BY-4.0 协议，仅用于学术研究。

用法:
    ./.venv/Scripts/python.exe scripts/download_travelplanner.py
"""

import sys
import urllib.request
from pathlib import Path

BASE = "https://hf-mirror.com/datasets/osunlp/TravelPlanner/resolve/main"
FILES = [
    "train.csv",
    "validation.csv",
    "test.csv",
    "example_submission.jsonl",
    "train_ref_info.jsonl",
    "validation_ref_info.jsonl",
    "test_ref_info.jsonl",
]


def download(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import ssl
    # 绕过 Windows schannel 的吊销检查问题
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for fname in FILES:
        out_path = out_dir / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [skip] {fname} 已存在 ({out_path.stat().st_size} bytes)")
            continue
        url = f"{BASE}/{fname}"
        print(f"  [download] {fname} <- {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=180) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            print(f"    → {len(data)} bytes")
        except Exception as e:
            print(f"    ✗ 失败: {e}", file=sys.stderr)


def main():
    out_dir = Path("data/external/travelplanner")
    print(f"下载 TravelPlanner 数据集到 {out_dir}")
    print(f"镜像: {BASE}")
    download(out_dir)
    print("\n完成。数据说明见 data/external/travelplanner/README.md")


if __name__ == "__main__":
    main()
