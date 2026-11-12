"""Semantic segmentation metrics (Toronto-3D protocol: labels 1..8, ignore 0)."""
from __future__ import annotations

import numpy as np

from .config import CLASS_NAMES, EVAL_CLASSES


def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """8x8 confusion matrix (rows = reference, cols = prediction) over labels 1..8."""
    m = (gt > 0) & (pred > 0)
    g = gt[m].astype(np.int64) - 1
    p = pred[m].astype(np.int64) - 1
    return np.bincount(g * 8 + p, minlength=64).reshape(8, 8)


def scores(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    present = cm.sum(1) > 0
    out = {
        "OA": 100 * tp.sum() / max(cm.sum(), 1),
        "mIoU": 100 * np.nanmean(np.where(present, iou, np.nan)) if present.any() else float("nan"),
    }
    for i, c in enumerate(EVAL_CLASSES):
        out[f"IoU_{CLASS_NAMES[c]}"] = 100 * iou[i] if present[i] else float("nan")
    return out


def evaluate(gt, pred) -> tuple[dict, np.ndarray]:
    cm = confusion(gt, pred)
    return scores(cm), cm


def mlflow_safe(d: dict, prefix: str = "") -> dict:
    """MLflow metric names cannot contain spaces in some backends; also drop NaNs."""
    return {prefix + k.replace(" ", "_"): float(v) for k, v in d.items() if v == v}
