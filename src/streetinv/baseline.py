"""Conventional baseline: handcrafted geometric features + Random Forest.

    python -m streetinv.baseline [--run-name rf_baseline]
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier

from .config import CLASS_NAMES, grid_dir, load_config, resolve
from .features import compute_features, feature_names
from .io import load_tile
from .metrics import evaluate, mlflow_safe
from .tracking import confusion_figure, save_prediction, setup_mlflow


def balanced_sample(labels, per_class, rng):
    idx = []
    for c in range(1, 9):
        cand = np.flatnonzero(labels == c)
        if len(cand):
            idx.append(rng.choice(cand, min(per_class, len(cand)), replace=False))
    return np.concatenate(idx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-name", default="rf_baseline")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    bc = cfg["baseline"]
    ks = tuple(bc["k_neighbours"])
    gdir = grid_dir(cfg)
    meta = json.loads((gdir / "meta.json").read_text())
    rng = np.random.default_rng(a.seed)

    mlflow = setup_mlflow(cfg)
    with mlflow.start_run(run_name=a.run_name) as run:
        mlflow.set_tags({"family": "conventional", "model": "RandomForest", "features": "handcrafted"})
        mlflow.log_params({"grid_size": cfg["data"]["grid_size"], "k_neighbours": str(ks),
                           "samples_per_class": bc["samples_per_class"], "n_estimators": bc["n_estimators"],
                           "max_depth": bc["max_depth"], "train_tiles": ",".join(cfg["data"]["train_tiles"]),
                           "test_tiles": ",".join(cfg["data"]["test_tiles"])})
        X, y = [], []
        t0 = time.time()
        for t in cfg["data"]["train_tiles"]:
            d = load_tile(gdir, t)
            idx = balanced_sample(d["label"], bc["samples_per_class"], rng)
            X.append(compute_features(d["xyz"], d["rgb"], d["intensity"], meta["tiles"][t]["intensity_p99"],
                                      ks, index=idx))
            y.append(d["label"][idx])
            print(f"{t}: {len(idx):,} training samples")
            del d
        X, y = np.concatenate(X), np.concatenate(y)
        clf = RandomForestClassifier(n_estimators=bc["n_estimators"], max_depth=bc["max_depth"],
                                     n_jobs=-1, class_weight="balanced", random_state=a.seed)
        clf.fit(X, y)
        mlflow.log_metric("train_seconds", time.time() - t0)
        imp = dict(sorted(zip(feature_names(ks), clf.feature_importances_.round(4).tolist()),
                          key=lambda kv: -kv[1]))
        mlflow.log_dict(imp, "feature_importance.json")
        model_path = resolve("models") / f"{a.run_name}.joblib"
        joblib.dump(clf, model_path, compress=3)
        del X, y

        for t in cfg["data"]["test_tiles"]:
            d = load_tile(gdir, t)
            tree = cKDTree(d["xyz"])
            t1 = time.time()
            pred = np.empty(len(d["xyz"]), np.uint8)
            conf = np.empty(len(d["xyz"]), np.float32)
            step = 300_000
            for s in range(0, len(pred), step):
                ii = np.arange(s, min(len(pred), s + step))
                F = compute_features(d["xyz"], d["rgb"], d["intensity"], meta["tiles"][t]["intensity_p99"],
                                     ks, index=ii, tree=tree)
                p = clf.predict_proba(F)
                pred[ii] = clf.classes_[p.argmax(1)]
                conf[ii] = p.max(1)
            m, cm = evaluate(d["label"], pred)
            mlflow.log_metric(f"{t}_infer_seconds", time.time() - t1)
            mlflow.log_metrics(mlflow_safe(m, f"{t}_"))
            mlflow.log_figure(confusion_figure(cm, f"RF baseline - {t}"), f"confusion_{t}.png")
            save_prediction(cfg, a.run_name, t, pred, conf, m, cm,
                            {"mlflow_run_id": run.info.run_id, "family": "conventional"})
            print(f"{t}: OA {m['OA']:.2f}  mIoU {m['mIoU']:.2f}")
            for c in range(1, 9):
                print(f"   {CLASS_NAMES[c]:>13s}: {m['IoU_' + CLASS_NAMES[c]]:.2f}")


if __name__ == "__main__":
    main()
