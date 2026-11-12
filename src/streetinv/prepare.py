"""Raw Toronto-3D PLY tiles -> voxel-subsampled .npz tiles.

    python -m streetinv.prepare [--config configs/default.yaml] [--grid 0.1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, grid_dir, load_config, resolve
from .io import grid_subsample_ply


def find_tiles(raw_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(raw_dir.rglob("L00*.ply"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--grid", type=float, default=None, help="voxel size in metres")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ov = {"data": {}}
    if a.grid:
        ov["data"]["grid_size"] = a.grid
    if a.raw_dir:
        ov["data"]["raw_dir"] = a.raw_dir
    cfg = load_config(a.config, ov)

    raw = resolve(cfg["data"]["raw_dir"])
    tiles = find_tiles(raw)
    if not tiles:
        raise SystemExit(f"No L00*.ply found under {raw}. Run scripts/download_toronto3d.py first.")
    out = grid_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    offset = np.array(cfg["data"]["utm_offset"], np.float64)
    meta = {"grid_size": cfg["data"]["grid_size"], "utm_offset": offset.tolist(),
            "crs": cfg["data"]["crs"], "tiles": {}}
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta["tiles"] = json.loads(meta_path.read_text()).get("tiles", {})

    for name, path in tiles.items():
        dst = out / f"{name}.npz"
        if dst.exists() and not a.force and name in meta["tiles"]:
            print(f"{name}: exists, skipping (use --force)")
            continue
        d = grid_subsample_ply(path, cfg["data"]["grid_size"], offset)
        np.savez(dst, **d)
        counts = np.bincount(d["label"], minlength=9)
        meta["tiles"][name] = {
            "n_raw": int(d["n_raw"]), "n_sub": int(len(d["xyz"])),
            "bounds_min": d["xyz"].min(0).tolist(), "bounds_max": d["xyz"].max(0).tolist(),
            "intensity_p99": float(np.nanpercentile(d["intensity"], 99)),
            "class_counts": {CLASS_NAMES[i]: int(c) for i, c in enumerate(counts)},
            "source": str(path),
        }
        print(f"{name}: {d['n_raw']:,} -> {len(d['xyz']):,} points")
        meta_path.write_text(json.dumps(meta, indent=2))
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
