"""Print a Markdown results table (segmentation + inventory) for the README.

    python -m streetinv.report [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import json

from .config import CLASS_NAMES, EVAL_CLASSES, inventory_dir, load_config, predictions_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    pdir = predictions_dir(cfg)
    cls = [CLASS_NAMES[c] for c in EVAL_CLASSES]
    for tile in cfg["data"]["test_tiles"]:
        print(f"### {tile}\n")
        print("| Model | OA | mIoU | " + " | ".join(cls) + " | Pole F1 | Car F1 |")
        print("|---" * (len(cls) + 5) + "|")
        for run in sorted(p.name for p in pdir.iterdir() if (p / f"{tile}_metrics.json").exists()):
            m = json.loads((pdir / run / f"{tile}_metrics.json").read_text())["metrics"]
            sf = inventory_dir(cfg) / run / f"{tile}_summary.json"
            inv = json.loads(sf.read_text()) if sf.exists() else {}
            f1 = [f"{inv[c]['F1']:.2f}" if c in inv else "-" for c in ("pole", "car")]
            print(f"| {run} | {m['OA']:.1f} | **{m['mIoU']:.1f}** | "
                  + " | ".join(f"{m['IoU_' + c]:.1f}" for c in cls) + f" | {f1[0]} | {f1[1]} |")
        print()


if __name__ == "__main__":
    main()
