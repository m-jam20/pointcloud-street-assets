"""Render README figures from the saved predictions -> reports/figures/*.png

    python scripts/make_figures.py [--tile L002] [--runs rf_baseline randla]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from streetinv.config import (CLASS_COLORS, CLASS_NAMES, EVAL_CLASSES, REFERENCE_RESULTS, ROOT, grid_dir,
                              inventory_dir, load_config, predictions_dir)
from streetinv.io import load_tile

OUT = ROOT / "reports" / "figures"
LABEL = {"rf_baseline": "Random Forest", "randla": "RandLA-Net"}
RUN_COLOR = {"rf_baseline": "#eb6834", "randla": "#2a78d6"}
INK, MUTED, GRID = "#1f1f1e", "#6f6e69", "#e6e5e0"
STATUS = {"TP": ("#1f9d55", "o", "correct"), "FP": ("#d03b3b", "x", "false alarm"), "FN": ("#eda100", "^", "missed")}
CREDIT = "Data: Toronto-3D (Tan et al. 2020), CC BY-NC 4.0"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white"})


def iou_chart(cfg, tile, runs):
    cls = [CLASS_NAMES[c] for c in EVAL_CLASSES]
    rows = {LABEL.get(r, r): json.loads((predictions_dir(cfg) / r / f"{tile}_metrics.json").read_text())["metrics"]
            for r in runs}
    paper = REFERENCE_RESULTS["RandLA-Net (paper, XYZ+RGB)"]
    names = ["mIoU"] + cls
    series = [(LABEL.get(r, r), [rows[LABEL.get(r, r)]["mIoU"]] + [rows[LABEL.get(r, r)][f"IoU_{c}"] for c in cls],
               RUN_COLOR.get(r, "#6250d6")) for r in runs]
    series.append(("RandLA-Net (paper)", [paper["mIoU"]] + [paper[c] for c in cls], "#c9c8c2"))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.8 / len(series)
    x = np.arange(len(names))
    for i, (lab, vals, col) in enumerate(series):
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.92, color=col, label=lab, zorder=2)
        if i < len(runs):  # label our runs only
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7, color=INK)
    ax.set_xticks(x, names, rotation=0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("IoU (%)")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.axvline(0.5, color=GRID, lw=1)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title(f"Semantic segmentation on test tile {tile}", loc="left", color=INK, pad=28)
    fig.text(0.99, -0.02, CREDIT, ha="right", fontsize=7, color=MUTED)
    fig.savefig(OUT / "iou_per_class.png")
    plt.close(fig)


def pick_window(xyz, gt, size):
    """Window with the most reference pole + car points (an interesting street section)."""
    m = (gt == 6) | (gt == 7)
    ij = np.floor(xyz[m, :2] / size).astype(np.int64)
    keys, cnt = np.unique(ij, axis=0, return_counts=True)
    c = (keys[cnt.argmax()] + 0.5) * size
    return c


def scene_3d(cfg, d, preds, tile, runs, size=35.0, max_pts=120_000):
    xyz, gt = d["xyz"], d["label"]
    c = pick_window(xyz, gt, size)
    m = np.flatnonzero((np.abs(xyz[:, 0] - c[0]) <= size / 2) & (np.abs(xyz[:, 1] - c[1]) <= size / 2))
    m = np.random.default_rng(0).choice(m, min(len(m), max_pts), replace=False)
    p = xyz[m] - np.r_[c, xyz[m, 2].min()]
    panels = [("Reference", gt[m])] + [(LABEL.get(r, r), preds[r][m]) for r in runs]
    fig = plt.figure(figsize=(5.2 * len(panels), 4.6))
    for i, (title, lab) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        cols = np.array([CLASS_COLORS[int(v)] for v in range(9)])[lab]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=cols, s=0.9, linewidths=0, depthshade=False)
        ax.set_box_aspect((np.ptp(p[:, 0]), np.ptp(p[:, 1]), np.ptp(p[:, 2])), zoom=1.65)
        ax.view_init(elev=38, azim=-62)
        ax.set_axis_off()
        txt = title
        if title != "Reference":
            ok = (gt[m] > 0)
            txt += f"  ({100 * (lab[ok] == gt[m][ok]).mean():.0f}% correct)"
        ax.set_title(txt, color=INK, fontsize=12)
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=CLASS_COLORS[k], label=CLASS_NAMES[k]) for k in range(1, 9)]
    handles.append(Line2D([], [], ls="", marker="o", ms=7, color=CLASS_COLORS[0], label="unclassified"))
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.99, -0.04, f"{size:.0f} m x {size:.0f} m section of {tile}. {CREDIT}", ha="right", fontsize=7, color=MUTED)
    fig.subplots_adjust(wspace=0, left=0, right=1, top=0.92, bottom=0.06)
    fig.savefig(OUT / "scene_comparison.png")
    plt.close(fig)


def inventory_map(cfg, d, tile, runs):
    xyz = d["xyz"]
    bg = np.random.default_rng(0).choice(len(xyz), min(len(xyz), 200_000), replace=False)
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 7), sharex=True, sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        idir = inventory_dir(cfg) / r
        objs = pd.read_csv(idir / f"{tile}_objects.csv")
        refs = pd.read_csv(idir / f"{tile}_reference.csv")
        s = json.loads((idir / f"{tile}_summary.json").read_text())["pole"]
        ax.scatter(xyz[bg, 0], xyz[bg, 1], s=0.05, c="#d9d8d3", linewidths=0, rasterized=True)
        pts = pd.concat([objs[objs["class"] == "pole"], refs[(refs["class"] == "pole") & (refs.status == "FN")]])
        for st, (col, mk, name) in STATUS.items():
            q = pts[pts.status == st]
            ax.scatter(q.x, q.y, s=26, c=col, marker=mk, linewidths=1.4 if mk == "x" else 0.6,
                       edgecolors="white" if mk != "x" else None, label=f"{name} ({len(q)})", zorder=3)
        ax.set_aspect("equal")
        ax.set_title(f"{LABEL.get(r, r)}: pole F1 {s['F1']:.2f}\nprecision {s['precision']:.2f} · recall {s['recall']:.2f}",
                     loc="left", color=INK, fontsize=11)
        ax.legend(frameon=False, loc="upper left", fontsize=8, markerscale=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.99, 0.01, f"Poles on test tile {tile} ({s['n_ref']} in reference). {CREDIT}", ha="right",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "pole_inventory.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tile", default="L002")
    ap.add_argument("--runs", nargs="+", default=["rf_baseline", "randla"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_tile(grid_dir(cfg), a.tile)
    preds = {r: np.load(predictions_dir(cfg) / r / f"{a.tile}.npz")["pred"] for r in a.runs}
    iou_chart(cfg, a.tile, a.runs)
    scene_3d(cfg, d, preds, a.tile, a.runs)
    inventory_map(cfg, d, a.tile, a.runs)
    for f in sorted(OUT.glob("*.png")):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
