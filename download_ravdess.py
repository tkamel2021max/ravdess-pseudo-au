"""Download RAVDESS video zips from Zenodo (record 1188976, CC BY-NC-SA 4.0) and unzip them.

Examples:
    python download_ravdess.py --out data/raw                    # 24 actors, speech videos
    python download_ravdess.py --out data/raw --actors 1 2 3 4   # subset (smoke test)
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/1188976/files/{name}?download=1"


def fetch(name, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name}")
        return
    tmp = dest.with_suffix(".part")

    def hook(b, bs, total):
        if total > 0:
            sys.stdout.write(f"\r{name}: {min(100, b * bs * 100 // total)}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL.format(name=name), tmp, hook)
    tmp.rename(dest)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--actors", type=int, nargs="*", default=list(range(1, 25)))
    ap.add_argument("--song", action="store_true", help="also download the singing videos")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["Speech"] + (["Song"] if a.song else [])
    for k in kinds:
        for i in a.actors:
            name = f"Video_{k}_Actor_{i:02d}.zip"
            z = out / name
            fetch(name, z)
            with zipfile.ZipFile(z) as f:
                f.extractall(out / f"{k.lower()}")
    print("done ->", out)


if __name__ == "__main__":
    main()
