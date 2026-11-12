"""Tile store, class-balanced crop sampler and the RandLA-Net index pyramid."""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .features import height_rasters
from .io import load_tile

FEATURE_NAMES = ["x", "y", "z", "r", "g", "b", "intensity", "height_above_ground"]
D_IN = len(FEATURE_NAMES)


class Tile:
    def __init__(self, gdir, name, meta):
        d = load_tile(gdir, name)
        self.name = name
        self.xyz = d["xyz"].astype(np.float32)
        ground, _ = height_rasters(self.xyz)
        p99 = max(meta["tiles"][name]["intensity_p99"], 1e-6)
        self.feats = np.concatenate([
            d["rgb"].astype(np.float32) / 255.0,
            np.clip(d["intensity"] / p99, 0, 1.5)[:, None],
            np.clip((self.xyz[:, 2] - ground) / 10.0, -1, 4)[:, None],
        ], 1).astype(np.float16)  # float16 halves RAM; cast back per crop
        self.label = d["label"].astype(np.uint8)
        del d, ground
        self.tree = cKDTree(self.xyz, balanced_tree=False)
        self.class_idx = {c: np.flatnonzero(self.label == c).astype(np.int32) for c in range(1, 9)}


def available_ram_mb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def load_tiles(gdir, names):
    meta = json.loads((gdir / "meta.json").read_text())
    return [Tile(gdir, n, meta) for n in names]


def build_pyramid(xyz: np.ndarray, k: int = 16, ratios=(4, 4, 4, 4)) -> dict:
    """xyz must be in random order: sub-sampling keeps the first N/r points of each level."""
    out = {"xyz": [], "neigh": [], "pool": [], "up": []}
    cur = xyz
    for r in ratios:
        nb = cKDTree(cur).query(cur, k=k)[1].astype(np.int32)
        n_sub = len(cur) // r
        sub = cur[:n_sub]
        up = cKDTree(sub).query(cur, k=1)[1].astype(np.int32)[:, None]
        out["xyz"].append(cur.astype(np.float32))
        out["neigh"].append(nb)
        out["pool"].append(nb[:n_sub])
        out["up"].append(up)
        cur = sub
    return out


def make_crop(tile: Tile, center_idx: int, num_points: int, k: int, rng=None, augment=False):
    """Return (sample dict of numpy arrays, original point indices, squared distances)."""
    n = min(num_points, len(tile.xyz))
    d2, idx = tile.tree.query(tile.xyz[center_idx], k=n)
    idx = np.atleast_1d(idx)
    d2 = np.atleast_1d(d2) ** 2
    perm = (rng or np.random.default_rng()).permutation(n)
    idx, d2 = idx[perm], d2[perm]
    xyz = tile.xyz[idx] - tile.xyz[center_idx]
    if augment:
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
        xyz = xyz @ R.T
        xyz *= rng.uniform(0.9, 1.1)
        if rng.random() < 0.5:
            xyz[:, 0] = -xyz[:, 0]
        xyz += rng.normal(0, 0.005, xyz.shape).astype(np.float32)
    xyz = xyz.astype(np.float32)
    sample = build_pyramid(xyz, k)
    sample["feats"] = np.concatenate([xyz, tile.feats[idx].astype(np.float32)], 1)
    sample["labels"] = tile.label[idx].astype(np.int64) - 1  # 0..7, unclassified -> -1 (ignored)
    return sample, idx, d2


def to_torch(sample: dict) -> dict:
    out = {}
    for key, v in sample.items():
        if isinstance(v, list):
            out[key] = [torch.from_numpy(np.ascontiguousarray(a)) for a in v]
        else:
            out[key] = torch.from_numpy(np.ascontiguousarray(v))
    return out


def batch_to_device(b: dict, device) -> dict:
    out = {}
    for key, v in b.items():
        if isinstance(v, list):
            out[key] = [(t.long() if t.dtype == torch.int32 else t).to(device, non_blocking=True) for t in v]
        else:
            out[key] = v.to(device, non_blocking=True)
    return out


class CropDataset(Dataset):
    """Class-balanced random crops: pick a class uniformly, then a random point of that class as centre."""

    def __init__(self, tiles, num_points, k, length, augment=True):
        self.tiles, self.num_points, self.k, self.length, self.augment = tiles, num_points, k, length, augment
        self.classes = [c for c in range(1, 9) if any(len(t.class_idx[c]) for t in tiles)]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        c = self.classes[rng.integers(len(self.classes))]
        w = np.array([len(t.class_idx[c]) for t in self.tiles], np.float64)
        t = self.tiles[rng.choice(len(self.tiles), p=w / w.sum())]
        center = t.class_idx[c][rng.integers(len(t.class_idx[c]))]
        sample, _, _ = make_crop(t, center, self.num_points, self.k, rng, self.augment)
        return to_torch(sample)


def class_weights(tiles) -> np.ndarray:
    """Inverse sqrt frequency (RandLA-Net), over labels 1..8."""
    cnt = np.zeros(8)
    for t in tiles:
        cnt += np.bincount(t.label, minlength=9)[1:9]
    freq = cnt / max(cnt.sum(), 1)
    w = 1.0 / (np.sqrt(freq) + 0.02)
    return (w / w.mean()).astype(np.float32)
