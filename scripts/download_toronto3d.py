"""Get Toronto-3D into data/raw/Toronto_3D/.

The dataset is hosted on OneDrive (https://github.com/WeikaiTan/Toronto-3D#download),
which does not allow scripted anonymous downloads. So:

  1. Open https://1drv.ms/u/s!Amlc6yZnF87psX6hKS8VOQllVvj4?e=yWhrYX in a browser
     and download Toronto_3D.zip (about 1.1 GB).
  2. Run this script pointing at the zip, e.g. from WSL:
        python scripts/download_toronto3d.py --zip /mnt/c/Users/<you>/Downloads/Toronto_3D.zip
     or, if you have a direct URL:
        python scripts/download_toronto3d.py --url <direct-download-url>

The data is licensed CC BY-NC 4.0 - it is never committed to git (see .gitignore).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEST = RAW / "Toronto_3D"
TILES = ["L001", "L002", "L003", "L004"]
SHARE_LINK = "https://1drv.ms/u/s!Amlc6yZnF87psX6hKS8VOQllVvj4?e=yWhrYX"


def have_tiles() -> list[str]:
    return sorted(p.stem for p in DEST.rglob("L00*.ply")) if DEST.exists() else []


def extract(zip_path: Path):
    print(f"extracting {zip_path} -> {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            name = Path(m.filename).name
            if m.is_dir() or not name or name.startswith("."):
                continue
            target = DEST / name  # flatten, and never write outside DEST
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            print(f"  {name}  ({target.stat().st_size / 2**20:.0f} MB)")


def download(url: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "Toronto_3D.zip"
    print(f"downloading {url} -> {out}")
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:.0f}/{total / 2**20:.0f} MB", end="")
    print()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="path to a downloaded Toronto_3D.zip")
    ap.add_argument("--url", help="direct download URL of Toronto_3D.zip")
    a = ap.parse_args()

    zip_path = a.zip
    if a.url:
        zip_path = download(a.url)
    if zip_path is None and (RAW / "Toronto_3D.zip").exists():
        zip_path = RAW / "Toronto_3D.zip"
    if zip_path is not None:
        if not zipfile.is_zipfile(zip_path):
            sys.exit(f"{zip_path} is not a valid zip (an HTML page instead of the file?)")
        extract(zip_path)

    found = have_tiles()
    missing = [t for t in TILES if t not in found]
    if missing:
        print(f"\nMissing tiles: {missing}\n\nDownload Toronto_3D.zip in a browser from:\n  {SHARE_LINK}\n"
              f"then run:\n  python scripts/download_toronto3d.py --zip <path/to/Toronto_3D.zip>")
        sys.exit(1)
    print(f"OK - tiles present in {DEST}: {found}\nNext: python -m streetinv.prepare")


if __name__ == "__main__":
    main()
