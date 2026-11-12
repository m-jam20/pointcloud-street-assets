"""Fast unit tests (no dataset download needed)."""
import numpy as np
import pandas as pd
import torch

from streetinv.dataset import build_pyramid
from streetinv.features import compute_features, feature_names
from streetinv.inventory import extract_instances, match
from streetinv.io import _reduce, _voxel_keys
from streetinv.metrics import evaluate
from streetinv.randla import RandLANet


def test_metrics_perfect_and_ignore():
    gt = np.array([0, 1, 2, 3, 6, 6, 7])
    m, cm = evaluate(gt, gt.copy())
    assert m["OA"] == 100 and m["mIoU"] == 100
    assert cm.sum() == 6  # label 0 ignored


def test_voxel_merge_matches_single_pass():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 5, (5000, 3))
    ones = np.ones(len(xyz))
    z3 = np.zeros((len(xyz), 3)); z1 = np.zeros(len(xyz)); z9 = np.zeros((len(xyz), 9))
    full = _reduce(_voxel_keys(xyz, 0.5), ones, xyz, z3, z1, z9)
    a = _reduce(_voxel_keys(xyz[:2000], 0.5), ones[:2000], xyz[:2000], z3[:2000], z1[:2000], z9[:2000])
    b = _reduce(_voxel_keys(xyz[2000:], 0.5), ones[2000:], xyz[2000:], z3[2000:], z1[2000:], z9[2000:])
    merged = _reduce(*[np.concatenate([x, y]) for x, y in zip(a, b)])
    assert np.array_equal(full[0], merged[0]) and np.allclose(full[2], merged[2])


def test_randla_forward_shapes():
    n, k, b = 1024, 8, 2
    batch = {"xyz": [], "neigh": [], "pool": [], "up": []}
    samples = [build_pyramid(np.random.rand(n, 3).astype(np.float32), k) for _ in range(b)]
    for key in batch:
        batch[key] = [torch.stack([torch.from_numpy(s[key][i]).long() if key != "xyz"
                                   else torch.from_numpy(s[key][i]) for s in samples]) for i in range(4)]
    batch["feats"] = torch.rand(b, n, 8)
    out = RandLANet(8, 8)(batch)
    assert out.shape == (b, n, 8)


def test_features_shape():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 10, (2000, 3)).astype(np.float32)
    f = compute_features(xyz, np.zeros((2000, 3), np.uint8), np.ones(2000, np.float32), 1.0, ks=(10, 20))
    assert f.shape == (2000, len(feature_names((10, 20)))) and np.isfinite(f).all()


def test_inventory_finds_poles_and_matches():
    rng = np.random.default_rng(0)
    pts, lab = [], []
    for x in (0.0, 10.0, 20.0):
        p = np.c_[rng.normal(x, 0.05, 300), rng.normal(0, 0.05, 300), rng.uniform(0, 6, 300)]
        pts.append(p); lab.append(np.full(300, 6))
    xyz = np.concatenate(pts); lab = np.concatenate(lab)
    inst = extract_instances(xyz, lab, 6, np.zeros(len(xyz)), cell=0.25, min_points=15, min_height=1.5)
    assert len(inst) == 3 and (inst.height > 5).all()
    pred = inst.drop(index=2).reset_index(drop=True)
    pred = pd.concat([pred, pd.DataFrame([{**inst.iloc[0].to_dict(), "x": 50.0}])], ignore_index=True)
    _, _, s = match(pred, inst, 1.0)
    assert (s["TP"], s["FP"], s["FN"]) == (2, 1, 1)
