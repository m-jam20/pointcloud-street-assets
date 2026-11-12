"""Generate small synthetic street tiles in the Toronto-3D PLY layout (for smoke tests / CI).

    python scripts/make_synthetic.py --out data/raw/synthetic --length 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

OFFSET = np.array([627285.0, 4841948.0, 0.0])


def street(rng, length=120.0, density=1.0):
    P, C, I, L = [], [], [], []

    def add(xyz, rgb, inten, lab):
        P.append(xyz); C.append(np.broadcast_to(rgb, xyz.shape).copy())
        I.append(np.full(len(xyz), inten) + rng.normal(0, 3, len(xyz))); L.append(np.full(len(xyz), lab))

    def n(x):
        return max(1, int(x * density))

    # road and sidewalks
    m = n(length * 24 * 40)
    xy = np.c_[rng.uniform(0, length, m), rng.uniform(-12, 12, m)]
    z = 0.02 * np.sin(xy[:, 0] / 15) + np.where(np.abs(xy[:, 1]) > 7, 0.15, 0) + rng.normal(0, 0.01, m)
    road = np.c_[xy, z]
    mark = (np.abs(xy[:, 1]) < 0.12) & ((xy[:, 0] % 6) < 3) | (np.abs(np.abs(xy[:, 1]) - 6.8) < 0.1)
    add(road[~mark], (85, 85, 90), 40, 1)
    add(road[mark], (235, 235, 225), 160, 2)
    # buildings
    for side in (-1, 1):
        x0 = 0.0
        while x0 < length:
            w, h = rng.uniform(8, 20), rng.uniform(6, 18)
            m = n(w * h * 30)
            pts = np.c_[rng.uniform(x0, x0 + w, m), np.full(m, side * rng.uniform(14, 15)), rng.uniform(0, h, m)]
            add(pts, tuple(rng.integers(90, 200, 3)), 70, 4)
            x0 += w + rng.uniform(0, 4)
    # poles, trees, cars, fences
    poles = np.arange(5, length, 25.0)
    for side in (-1, 1):
        for px in poles + (side > 0) * 12:
            h = rng.uniform(6, 9)
            m = n(h * 120)
            a = rng.uniform(0, 2 * np.pi, m)
            pts = np.c_[px + 0.12 * np.cos(a), side * 9 + 0.12 * np.sin(a), rng.uniform(0, h, m)]
            arm = np.c_[np.full(n(80), px), side * 9 - side * rng.uniform(0, 2, n(80)), np.full(n(80), h)]
            add(np.r_[pts, arm], (120, 120, 130), 90, 6)
        for tx in np.arange(15, length, 18.0):
            c = np.array([tx + rng.uniform(-2, 2), side * 11, rng.uniform(5, 7)])
            m = n(2500)
            v = rng.normal(size=(m, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
            add(c + v * rng.uniform(1.5, 3, (m, 1)) * [1, 1, 0.8], (60, 120, 50), 50, 3)
            tr = n(200)
            add(np.c_[np.full(tr, c[0]), np.full(tr, c[1]), rng.uniform(0, c[2] - 2, tr)] + rng.normal(0, 0.1, (tr, 3)),
                (90, 70, 50), 45, 3)
        for cx in rng.uniform(0, length - 5, 4):
            m = n(1500)
            pts = np.c_[rng.uniform(cx, cx + 4.5, m), side * 5.5 + rng.uniform(-0.9, 0.9, m), rng.uniform(0.3, 1.5, m)]
            add(pts, tuple(rng.integers(0, 255, 3)), 120, 7)
        m = n(length * 1.0 * 40)
        add(np.c_[rng.uniform(0, length, m), np.full(m, side * 12.5), rng.uniform(0, 1.0, m)], (110, 100, 90), 60, 8)
        # utility line between poles
        x = rng.uniform(0, length, n(length * 30))
        sag = 0.4 * ((x - 5) % 25 / 25 - 0.5) ** 2 * 4
        add(np.c_[x, np.full(len(x), side * 9), 7.5 - sag], (40, 40, 40), 30, 5)
    # unclassified clutter
    m = n(2000)
    add(np.c_[rng.uniform(0, length, m), rng.uniform(-12, 12, m), rng.uniform(0, 3, m)], (150, 150, 150), 50, 0)
    xyz = np.concatenate(P) + rng.normal(0, 0.005, (sum(len(p) for p in P), 3))
    return xyz, np.concatenate(C), np.concatenate(I), np.concatenate(L)


def write_ply(path, xyz, rgb, inten, lab):
    v = np.empty(len(xyz), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("red", "u1"), ("green", "u1"),
                                  ("blue", "u1"), ("scalar_Intensity", "f4"), ("scalar_GPSTime", "f4"),
                                  ("scalar_ScanAngleRank", "f4"), ("scalar_Label", "f4")])
    xyz = xyz + OFFSET
    v["x"], v["y"], v["z"] = xyz.T
    v["red"], v["green"], v["blue"] = rgb.T
    v["scalar_Intensity"] = np.clip(inten, 0, None)
    v["scalar_GPSTime"] = 0
    v["scalar_ScanAngleRank"] = 0
    v["scalar_Label"] = lab
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--length", type=float, default=120.0)
    ap.add_argument("--density", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(["L001", "L002", "L003", "L004"]):
        xyz, rgb, inten, lab = street(np.random.default_rng(i), a.length, a.density)
        write_ply(out / f"{t}.ply", xyz, rgb, inten, lab)
        print(f"{t}: {len(xyz):,} points")


if __name__ == "__main__":
    main()
