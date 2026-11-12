"""Thin MLflow helpers + saving predictions in a common format for the app."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES, EVAL_CLASSES, ROOT, predictions_dir

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def setup_mlflow(cfg: dict):
    import mlflow

    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(ROOT / uri[len("sqlite:///"):])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    return mlflow


def confusion_figure(cm: np.ndarray, title: str) -> plt.Figure:
    names = [CLASS_NAMES[c] for c in EVAL_CLASSES]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8), names, rotation=45, ha="right")
    ax.set_yticks(range(8), names)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("reference")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def save_prediction(cfg: dict, run_name: str, tile: str, pred: np.ndarray, conf: np.ndarray | None,
                    metrics: dict, cm: np.ndarray, info: dict | None = None) -> Path:
    """predictions/<run_name>/<tile>.npz + <tile>_metrics.json — read by the Streamlit app."""
    d = predictions_dir(cfg) / run_name
    d.mkdir(parents=True, exist_ok=True)
    arrs = {"pred": pred.astype(np.uint8)}
    if conf is not None:
        arrs["conf"] = conf.astype(np.float16)
    np.savez_compressed(d / f"{tile}.npz", **arrs)
    payload = {"run_name": run_name, "tile": tile, "metrics": metrics, "confusion": cm.tolist(), **(info or {})}
    (d / f"{tile}_metrics.json").write_text(json.dumps(payload, indent=2))
    return d
