"""Handcrafted point features (the "conventional" pipeline, Weinmann et al. 2015 style).

Per point:
* eigenvalue features of the local covariance at several k (linearity, planarity,
  scattering, omnivariance, anisotropy, eigen-entropy, change of curvature),
  verticality, local z-range / z-std
* height above local ground, distance to local top (2-D raster)
* colour and intensity
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree


def _eigen_block(xyz, idx):
    nb = xyz[idx]                                   # (n, k, 3)
    c = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / idx.shape[1]
    w, v = np.linalg.eigh(cov)                       # ascending
    w = np.clip(w, 1e-12, None)
    l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2]
    s = l1 + l2 + l3
    e = w / s[:, None]
    normal_z = np.abs(v[:, 2, 0])
    zr = nb[:, :, 2].max(1) - nb[:, :, 2].min(1)
    return np.stack([
        (l1 - l2) / l1,                   # linearity
        (l2 - l3) / l1,                   # planarity
        l3 / l1,                          # scattering
        np.cbrt(l1 * l2 * l3) / s,        # omnivariance (scale-free)
        (l1 - l3) / l1,                   # anisotropy
        -(e * np.log(e)).sum(1),          # eigen-entropy
        l3 / s,                           # change of curvature
        1.0 - normal_z,                   # verticality
        zr,                               # local z range
        nb[:, :, 2].std(1),               # local z std
    ], 1)


EIGEN_NAMES = ["linearity", "planarity", "scattering", "omnivariance", "anisotropy",
               "eigenentropy", "curvature", "verticality", "z_range", "z_std"]


def height_rasters(xyz, cell=1.0, win=5):
    """Local ground (min z) and top (max z) from a 2-D grid, smoothed over win x win cells."""
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / cell).astype(np.int64)
    shape = tuple(ij.max(0) + 1)
    lin = ij[:, 0] * shape[1] + ij[:, 1]
    zmin = np.full(shape[0] * shape[1], np.inf, np.float32)
    zmax = np.full(shape[0] * shape[1], -np.inf, np.float32)
    np.minimum.at(zmin, lin, xyz[:, 2])
    np.maximum.at(zmax, lin, xyz[:, 2])
    zmin = minimum_filter(zmin.reshape(shape), size=win, mode="nearest").ravel()
    zmax = maximum_filter(zmax.reshape(shape), size=win, mode="nearest").ravel()
    return zmin[lin], zmax[lin]


def compute_features(xyz, rgb, intensity, intensity_scale, ks=(10, 30), index=None,
                     tree=None, chunk=100_000):
    """Feature matrix for xyz[index] (all points if index is None). Neighbours from the full cloud."""
    xyz = xyz.astype(np.float64)
    tree = tree or cKDTree(xyz)
    index = np.arange(len(xyz)) if index is None else np.asarray(index)
    ground, top = height_rasters(xyz)
    kmax = max(ks)
    out = []
    for s in range(0, len(index), chunk):
        ii = index[s:s + chunk]
        _, nn = tree.query(xyz[ii], k=kmax, workers=-1)
        f = [_eigen_block(xyz, nn[:, :k]) for k in ks]
        z = xyz[ii, 2]
        f.append(np.stack([z - ground[ii], top[ii] - z, top[ii] - ground[ii]], 1))
        f.append(rgb[ii].astype(np.float64) / 255.0)
        f.append((intensity[ii] / intensity_scale)[:, None])
        out.append(np.concatenate(f, 1).astype(np.float32))
    return np.concatenate(out, 0)


def feature_names(ks=(10, 30)):
    names = [f"{n}_k{k}" for k in ks for n in EIGEN_NAMES]
    return names + ["height_above_ground", "dist_to_top", "column_height", "r", "g", "b", "intensity"]
