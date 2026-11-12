"""Reading Toronto-3D PLY tiles and memory-safe voxel-grid sub-sampling.

A raw tile holds up to ~20 M points, which is too much for 8 GB of RAM if it
is loaded at once. We memory-map the PLY and voxelise it chunk by chunk,
merging the partial voxel sums at the end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData
from tqdm import tqdm

N_LABELS = 9


def _find(names, *candidates):
    low = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    for c in candidates:  # substring fallback, e.g. "scalar_Label"
        for n in names:
            if c.lower() in n.lower():
                return n
    return None


def open_ply(path: str | Path):
    """Return the vertex structured array (memory-mapped) and the field mapping."""
    ply = PlyData.read(str(path), mmap="r")
    v = ply["vertex"].data
    names = v.dtype.names
    fields = {
        "x": _find(names, "x"), "y": _find(names, "y"), "z": _find(names, "z"),
        "r": _find(names, "red", "r"), "g": _find(names, "green", "g"), "b": _find(names, "blue", "b"),
        "intensity": _find(names, "scalar_Intensity", "intensity"),
        "label": _find(names, "scalar_Label", "label", "class"),
    }
    missing = [k for k in ("x", "y", "z") if fields[k] is None]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}; found {names}")
    return v, fields


def read_chunk(v, fields, start, end, offset):
    c = v[start:end]
    xyz = np.stack([np.asarray(c[fields[a]], np.float64) - offset[i] for i, a in enumerate("xyz")], 1)
    if fields["r"]:
        rgb = np.stack([np.asarray(c[fields[a]]) for a in "rgb"], 1).astype(np.float32)
        if rgb.max(initial=0) > 255:  # 16-bit colours
            rgb /= 256.0
    else:
        rgb = np.zeros((len(c), 3), np.float32)
    inten = np.array(c[fields["intensity"]], np.float32) if fields["intensity"] else np.zeros(len(c), np.float32)
    bad = ~np.isfinite(inten)
    if bad.any():  # e.g. 0.19 % of Toronto-3D L003 has NaN intensity -> fill with the chunk median
        inten[bad] = np.median(inten[~bad]) if (~bad).any() else 0.0
    lab = np.asarray(c[fields["label"]]).astype(np.int64) if fields["label"] else np.zeros(len(c), np.int64)
    lab = np.clip(lab, 0, N_LABELS - 1)
    return xyz, rgb, inten, lab


def _voxel_keys(xyz, grid):
    ijk = np.floor(xyz / grid).astype(np.int64) + (1 << 20)
    if (ijk < 0).any() or (ijk >= (1 << 21)).any():
        raise ValueError("coordinates out of range for voxel keys - check utm_offset")
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def _reduce(keys, cnt, sxyz, srgb, sint, slab):
    """Merge rows with equal voxel keys. Compact dtypes keep RAM low (~70 B / voxel)."""
    uniq, inv = np.unique(keys, return_inverse=True)
    del keys
    n = len(uniq)
    out_cnt = np.bincount(inv, weights=cnt, minlength=n).astype(np.uint32)
    out_xyz = np.empty((n, 3), np.float64)
    out_rgb = np.empty((n, 3), np.float32)
    out_lab = np.empty((n, N_LABELS), np.uint16)
    for i in range(3):
        out_xyz[:, i] = np.bincount(inv, weights=sxyz[:, i], minlength=n)
        out_rgb[:, i] = np.bincount(inv, weights=srgb[:, i], minlength=n)
    out_int = np.bincount(inv, weights=sint, minlength=n).astype(np.float32)
    for i in range(N_LABELS):
        out_lab[:, i] = np.minimum(np.bincount(inv, weights=slab[:, i], minlength=n), 65535)
    return uniq, out_cnt, out_xyz, out_rgb, out_int, out_lab


def grid_subsample_ply(path, grid, offset, chunk=2_000_000, verbose=True):
    """Voxel-grid sub-sample a PLY tile chunk by chunk (memory-mapped), merging incrementally.

    Returns dict of arrays (xyz relative to offset). Peak RAM ~ a few hundred MB per chunk
    plus ~70 bytes per output voxel.
    """
    v, fields = open_ply(path)
    n = len(v)
    acc = None
    for s in tqdm(range(0, n, chunk), desc=f"voxelising {Path(path).name}", disable=not verbose):
        xyz, rgb, inten, lab = read_chunk(v, fields, s, min(n, s + chunk), offset)
        onehot = np.zeros((len(lab), N_LABELS), np.uint8)
        onehot[np.arange(len(lab)), lab] = 1
        part = _reduce(_voxel_keys(xyz, grid), np.ones(len(lab), np.uint8), xyz, rgb, inten, onehot)
        del xyz, rgb, inten, lab, onehot
        if acc is None:
            acc = part
        else:
            merged = [np.concatenate([a, b]) for a, b in zip(acc, part)]
            del acc, part
            acc = _reduce(*merged)
            del merged
    _, cnt, sxyz, srgb, sint, slab = acc
    del acc
    cnt_ = cnt[:, None].astype(np.float64)
    return {
        "xyz": (sxyz / cnt_).astype(np.float32),
        "rgb": np.clip(srgb / cnt_, 0, 255).astype(np.uint8),
        "intensity": (sint / cnt).astype(np.float32),
        "label": slab.argmax(1).astype(np.uint8),
        "n_raw": np.int64(n),
    }


def load_tile(grid_path: Path, name: str) -> dict:
    with np.load(grid_path / f"{name}.npz") as d:
        return {k: d[k] for k in d.files}
