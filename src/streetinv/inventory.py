"""Turn per-point labels into an object inventory (poles, cars) and score it.

Instances = 2-D connected components of the class points on a fine raster
(fast, memory-light, the usual GIS approach). Toronto-3D has no instance labels,
so the *reference* inventory is built the same way from the ground-truth labels.

    python -m streetinv.inventory --run randla [--tiles L002]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .config import CLASS_NAMES, grid_dir, inventory_dir, load_config, predictions_dir
from .features import height_rasters
from .io import load_tile

INVENTORY_CLASSES = {"pole": 6, "car": 7}


def extract_instances(xyz, labels, cls, ground, cell, min_points, min_height):
    m = labels == cls
    pts, g = xyz[m], ground[m]
    cols = ["obj_id", "class", "x", "y", "z_base", "height", "n_points", "extent_x", "extent_y", "footprint_m2"]
    if len(pts) == 0:
        return pd.DataFrame(columns=cols)
    mn = pts[:, :2].min(0)
    ij = np.floor((pts[:, :2] - mn) / cell).astype(np.int64)
    occ = np.zeros(tuple(ij.max(0) + 1), bool)
    occ[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    pid = lab[ij[:, 0], ij[:, 1]] - 1
    cnt = np.bincount(pid, minlength=n)
    cx = np.bincount(pid, pts[:, 0], n) / np.maximum(cnt, 1)
    cy = np.bincount(pid, pts[:, 1], n) / np.maximum(cnt, 1)
    zmax = np.full(n, -np.inf); np.maximum.at(zmax, pid, pts[:, 2])
    gmin = np.full(n, np.inf); np.minimum.at(gmin, pid, g)
    xmin = np.full(n, np.inf); np.minimum.at(xmin, pid, pts[:, 0])
    xmax = np.full(n, -np.inf); np.maximum.at(xmax, pid, pts[:, 0])
    ymin = np.full(n, np.inf); np.minimum.at(ymin, pid, pts[:, 1])
    ymax = np.full(n, -np.inf); np.maximum.at(ymax, pid, pts[:, 1])
    cells = np.bincount(lab[lab > 0] - 1, minlength=n)
    height = zmax - gmin
    keep = (cnt >= min_points) & (height >= min_height)
    df = pd.DataFrame({
        "class": CLASS_NAMES[cls], "x": cx, "y": cy, "z_base": gmin, "height": height, "n_points": cnt,
        "extent_x": xmax - xmin, "extent_y": ymax - ymin, "footprint_m2": cells * cell * cell,
    })[keep].reset_index(drop=True)
    df.insert(0, "obj_id", np.arange(len(df)))
    return df[cols]


def match(pred: pd.DataFrame, ref: pd.DataFrame, max_dist: float):
    pred = pred.copy()
    ref = ref.copy()
    pred["status"], pred["ref_id"], pred["match_dist"] = "FP", -1, np.nan
    ref["status"], ref["pred_id"] = "FN", -1
    if len(pred) and len(ref):
        d = np.hypot(pred.x.values[:, None] - ref.x.values[None], pred.y.values[:, None] - ref.y.values[None])
        r, c = linear_sum_assignment(np.where(d <= max_dist, d, 1e6))
        ok = d[r, c] <= max_dist
        r, c = r[ok], c[ok]
        pred.loc[r, "status"], pred.loc[r, "ref_id"], pred.loc[r, "match_dist"] = "TP", c, d[r, c]
        ref.loc[c, "status"], ref.loc[c, "pred_id"] = "TP", r
    tp = int((pred.status == "TP").sum())
    p = tp / len(pred) if len(pred) else float("nan")
    rcl = tp / len(ref) if len(ref) else float("nan")
    f1 = 2 * p * rcl / (p + rcl) if tp else 0.0
    mt = pred[pred.status == "TP"]
    h_err = (mt.height.values - ref.loc[mt.ref_id.values, "height"].values) if len(mt) else np.array([])
    stats = {"n_pred": len(pred), "n_ref": len(ref), "TP": tp, "FP": len(pred) - tp, "FN": len(ref) - tp,
             "precision": p, "recall": rcl, "F1": f1,
             "mean_pos_error_m": float(mt.match_dist.mean()) if len(mt) else float("nan"),
             "mean_abs_height_error_m": float(np.abs(h_err).mean()) if len(h_err) else float("nan")}
    return pred, ref, stats


def add_lonlat(df: pd.DataFrame, offset, crs):
    from pyproj import Transformer

    if not len(df):
        df["lon"], df["lat"] = [], []
        return df
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(df.x.values + offset[0], df.y.values + offset[1])
    df["lon"], df["lat"] = lon, lat
    return df


def to_geojson(df: pd.DataFrame) -> dict:
    feats = []
    for _, r in df.iterrows():
        props = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if k not in ("lon", "lat")}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def build_inventory(cfg, run: str, tile: str, d: dict | None = None):
    gdir = grid_dir(cfg)
    meta = json.loads((gdir / "meta.json").read_text())
    d = d or load_tile(gdir, tile)
    ground, _ = height_rasters(d["xyz"])
    pred_labels = np.load(predictions_dir(cfg) / run / f"{tile}.npz")["pred"]
    off, crs = meta["utm_offset"], meta["crs"]
    preds, refs, summary = [], [], {}
    for name, cls in INVENTORY_CLASSES.items():
        p = cfg["inventory"][name]
        kw = dict(cell=p["cell"], min_points=p["min_points"], min_height=p["min_height"])
        pi = extract_instances(d["xyz"], pred_labels, cls, ground, **kw)
        ri = extract_instances(d["xyz"], d["label"], cls, ground, **kw)
        pi, ri, st = match(pi, ri, p["match_dist"])
        preds.append(pi)
        refs.append(ri)
        summary[name] = st
    def cat(dfs):
        nonempty = [d for d in dfs if len(d)]
        return pd.concat(nonempty, ignore_index=True) if nonempty else dfs[0]

    pred_df = add_lonlat(cat(preds), off, crs)
    ref_df = add_lonlat(cat(refs), off, crs)
    out = inventory_dir(cfg) / run
    out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out / f"{tile}_objects.csv", index=False)
    ref_df.to_csv(out / f"{tile}_reference.csv", index=False)
    (out / f"{tile}_objects.geojson").write_text(json.dumps(to_geojson(pred_df)))
    (out / f"{tile}_summary.json").write_text(json.dumps(summary, indent=2))
    return pred_df, ref_df, summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True, help="prediction folder name, e.g. randla or rf_baseline")
    ap.add_argument("--tiles", nargs="+", default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    for tile in a.tiles or cfg["data"]["test_tiles"]:
        _, _, s = build_inventory(cfg, a.run, tile)
        for name, st in s.items():
            print(f"{tile} {name:5s}: pred {st['n_pred']:4d} ref {st['n_ref']:4d}  P {st['precision']:.3f} "
                  f"R {st['recall']:.3f} F1 {st['F1']:.3f}  pos.err {st['mean_pos_error_m']:.2f} m")
        mfile = predictions_dir(cfg) / a.run / f"{tile}_metrics.json"
        run_id = json.loads(mfile.read_text()).get("mlflow_run_id") if mfile.exists() else None
        if run_id and not a.no_mlflow:
            from .tracking import setup_mlflow

            mlflow = setup_mlflow(cfg)
            with mlflow.start_run(run_id=run_id):
                for name, st in s.items():
                    mlflow.log_metrics({f"{tile}_inv_{name}_{k}": float(v) for k, v in st.items() if v == v})


if __name__ == "__main__":
    main()
