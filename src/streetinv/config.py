"""Configuration, paths and class definitions."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

# Toronto-3D labels. 0 = unclassified, ignored in training and evaluation.
CLASS_NAMES = {
    0: "unclassified",
    1: "road",
    2: "road marking",
    3: "natural",
    4: "building",
    5: "utility line",
    6: "pole",
    7: "car",
    8: "fence",
}
EVAL_CLASSES = list(range(1, 9))
NUM_CLASSES = 8  # network output = labels 1..8 shifted to 0..7

CLASS_COLORS = {  # hex, semantic where possible (vegetation green, road grey); used by app + figures
    0: "#bdbcb6",  # unclassified - muted
    1: "#6b6b73",  # road
    2: "#eda100",  # road marking
    3: "#008300",  # natural
    4: "#eb6834",  # building
    5: "#6250d6",  # utility line
    6: "#2a78d6",  # pole
    7: "#e87ba4",  # car
    8: "#8c5a2b",  # fence
}

# Published reference (Toronto-3D README, RandLA-Net XYZ+RGB, Hu et al. 2021), for context.
REFERENCE_RESULTS = {
    "RandLA-Net (paper, XYZ+RGB)": {
        "OA": 94.37, "mIoU": 81.77, "road": 96.69, "road marking": 64.21, "natural": 96.92,
        "building": 94.24, "utility line": 88.06, "pole": 77.84, "car": 93.37, "fence": 42.86,
    },
    "KPConv (paper, XYZ)": {
        "OA": 95.39, "mIoU": 69.11, "road": 94.62, "road marking": 0.06, "natural": 96.07,
        "building": 91.51, "utility line": 87.68, "pole": 81.56, "car": 85.66, "fence": 15.72,
    },
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def grid_dir(cfg: dict) -> Path:
    return resolve(cfg["data"]["processed_dir"]) / f"grid_{cfg['data']['grid_size']:.2f}"


def predictions_dir(cfg: dict) -> Path:
    return grid_dir(cfg) / "predictions"


def inventory_dir(cfg: dict) -> Path:
    return grid_dir(cfg) / "inventory"
