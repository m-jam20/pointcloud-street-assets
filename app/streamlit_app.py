"""Street furniture inventory - interactive explorer.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from streetinv.config import (CLASS_COLORS, CLASS_NAMES, EVAL_CLASSES, REFERENCE_RESULTS,  # noqa: E402
                              grid_dir, inventory_dir, load_config, predictions_dir)
from streetinv.io import load_tile  # noqa: E402
from streetinv.metrics import evaluate  # noqa: E402

st.set_page_config(page_title="Street inventory · Toronto-3D", page_icon="🏙️", layout="wide")

# Fixed categorical order for runs (colour follows the run, never its rank)
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300"]
REF_GREY = "#a3a29c"
STATUS = {"TP": "#1f9d55", "FP": "#d03b3b", "FN": "#eda100"}
OK_GREY, ERR_RED = "#c9c8c2", "#d03b3b"
SEQ = [[0, "#cfe0f5"], [1, "#0b3d91"]]  # one-hue sequential ramp


# ----------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading tile…", max_entries=1)  # no copy per rerun
def get_tile(path: str, mtime: float) -> dict:
    p = Path(path)
    return load_tile(p.parent, p.stem)


@st.cache_resource(show_spinner=False, max_entries=8)
def get_pred(path: str, mtime: float) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def overview_index(n: int, k: int = 150_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n, min(n, k), replace=False))


@st.cache_data(show_spinner=False)
def dense_centre(path: str, mtime: float) -> tuple[float, float]:
    """Centre of the densest 20 m cell - always lands on the street, even for diagonal tiles."""
    xy = get_tile(path, mtime)["xyz"][::20, :2]
    ij = np.floor((xy - xy.min(0)) / 20).astype(np.int64)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    best = np.bincount(key).argmax()
    c = xy[key == best].mean(0)
    return float(round(c[0])), float(round(c[1]))


def subsample(idx: np.ndarray, k: int) -> np.ndarray:
    if len(idx) <= k:
        return idx
    return np.sort(np.random.default_rng(1).choice(idx, k, replace=False))


# ----------------------------------------------------------------------------- figures
def label_traces(x, y, z, labels, size, three_d=True, prefix=""):
    traces = []
    for c in range(9):
        m = labels == c
        if not m.any():
            continue
        kw = dict(mode="markers", name=f"{prefix}{CLASS_NAMES[c]}", legendgroup=str(c),
                  marker=dict(size=size, color=CLASS_COLORS[c]),
                  hovertemplate=f"{CLASS_NAMES[c]}<br>x %{{x:.1f}} y %{{y:.1f}}" + (" z %{z:.1f}" if three_d else "")
                  + "<extra></extra>")
        traces.append(go.Scatter3d(x=x[m], y=y[m], z=z[m], **kw) if three_d else go.Scattergl(x=x[m], y=y[m], **kw))
    return traces


def colour_spec(source: str, idx: np.ndarray, d: dict, preds: dict):
    """-> (kind, values, title)."""
    if source == "RGB":
        return "rgb", d["rgb"][idx], "RGB"
    if source == "Reference labels":
        return "labels", d["label"][idx], "Reference"
    if source == "Intensity":
        return "scalar", d["intensity"][idx], "Intensity"
    if source == "Height (z)":
        return "scalar", d["xyz"][idx, 2], "z (m)"
    run, what = source.split(" · ")
    p = preds[run]
    if what == "prediction":
        return "labels", p["pred"][idx], run
    if what == "confidence":
        return "scalar", p["conf"][idx].astype(np.float32) if "conf" in p else np.ones(len(idx)), "confidence"
    gt, pr = d["label"][idx], p["pred"][idx]
    return "errors", np.where(gt == 0, 2, (gt != pr).astype(np.int8)), run


def fig_3d(xyz, kind, vals, size, title, height=620, rev="3d"):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if kind == "labels":
        traces = label_traces(x, y, z, vals, size)
    elif kind == "errors":
        traces = []
        for code, name, col in ((0, "correct", OK_GREY), (1, "wrong", ERR_RED), (2, "unclassified (ignored)", "#e8e7e2")):
            m = vals == code
            if m.any():
                traces.append(go.Scatter3d(x=x[m], y=y[m], z=z[m], mode="markers", name=name,
                                           marker=dict(size=size + (1 if code == 1 else 0), color=col)))
    elif kind == "rgb":
        cols = [f"rgb({r},{g},{b})" for r, g, b in vals]
        traces = [go.Scatter3d(x=x, y=y, z=z, mode="markers", name="RGB", showlegend=False,
                               marker=dict(size=size, color=cols))]
    else:
        lo, hi = np.percentile(vals, [1, 99]) if len(vals) else (0, 1)
        traces = [go.Scatter3d(x=x, y=y, z=z, mode="markers", showlegend=False,
                               marker=dict(size=size, color=vals, colorscale=SEQ, cmin=lo, cmax=hi,
                                           colorbar=dict(title=title, thickness=12)))]
    fig = go.Figure(traces)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), title=dict(text=title, x=0.01, font_size=14),
                      scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                      legend=dict(itemsizing="constant", orientation="h", y=-0.02), uirevision=rev)
    return fig


def fig_topdown(xy, traces_fn, rect=None, height=330):
    fig = go.Figure(traces_fn())
    if rect is not None:
        fig.add_shape(type="rect", x0=rect[0], x1=rect[1], y0=rect[2], y1=rect[3],
                      line=dict(color="#0b0b0b", width=2), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), dragmode="select",
                      legend=dict(itemsizing="constant", orientation="h", y=-0.15),
                      xaxis=dict(title="x (m)", showgrid=False), yaxis=dict(title="y (m)", showgrid=False,
                                                                           scaleanchor="x"))
    return fig


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🏙️ Street inventory")
cfg_files = sorted((ROOT / "configs").glob("*.yaml"))
names = [c.name for c in cfg_files]
cfg_name = st.sidebar.selectbox("Dataset config", names, index=names.index("default.yaml") if "default.yaml" in names else 0)
cfg = load_config(ROOT / "configs" / cfg_name)
gdir = grid_dir(cfg)
if not (gdir / "meta.json").exists():
    st.title("No prepared data yet")
    st.markdown(f"Expected `{gdir.relative_to(ROOT)}/meta.json`.\n\n```bash\n"
                "python scripts/download_toronto3d.py --zip <Toronto_3D.zip>\npython -m streetinv.prepare\n```\n"
                "Or try the synthetic smoke-test data: `python scripts/make_synthetic.py` then "
                "`python -m streetinv.prepare --config configs/synthetic.yaml`.")
    st.stop()

meta = json.loads((gdir / "meta.json").read_text())
tiles = sorted(meta["tiles"])
default_tile = cfg["data"]["test_tiles"][0] if cfg["data"]["test_tiles"][0] in tiles else tiles[0]
tile = st.sidebar.selectbox("Tile", tiles, index=tiles.index(default_tile),
                            help="Official split: L002 = test, others = train")
is_test = tile in cfg["data"]["test_tiles"]
st.sidebar.caption(("🧪 test tile" if is_test else "🏋️ training tile - predictions here are optimistic") +
                   f" · {meta['tiles'][tile]['n_sub']:,} points @ {meta['grid_size']} m")

tpath = gdir / f"{tile}.npz"
d = get_tile(str(tpath), mtime(tpath))
pdir = predictions_dir(cfg)
all_runs = sorted(p.name for p in pdir.iterdir() if p.is_dir()) if pdir.exists() else []
run_color = {r: RUN_COLORS[i % len(RUN_COLORS)] for i, r in enumerate(all_runs)}
runs = [r for r in all_runs if (pdir / r / f"{tile}.npz").exists()]
preds = {r: get_pred(str(pdir / r / f"{tile}.npz"), mtime(pdir / r / f"{tile}.npz")) for r in runs}
preds = {r: p for r, p in preds.items() if len(p["pred"]) == len(d["xyz"])}
runs = list(preds)

# window (applied from a box selection before the widgets are built)
bmin, bmax = np.array(meta["tiles"][tile]["bounds_min"]), np.array(meta["tiles"][tile]["bounds_max"])
wid = f"{cfg_name}:{tile}"  # window state per dataset AND tile (tile names repeat across datasets)
kx, ky, ks = f"cx_{wid}", f"cy_{wid}", f"size_{wid}"
if kx not in st.session_state:
    c = dense_centre(str(tpath), mtime(tpath))
    st.session_state[kx], st.session_state[ky], st.session_state[ks] = c[0], c[1], 50
if (pw := st.session_state.pop("pending_window", None)) and pw[0] == wid:
    st.session_state[kx], st.session_state[ky], st.session_state[ks] = pw[1], pw[2], pw[3]

st.sidebar.subheader("3-D window")
st.sidebar.caption("Tip: box-select on any top-down map to move the window.")
cx = st.sidebar.slider("centre x (m)", float(np.floor(bmin[0])), float(np.ceil(bmax[0])), key=kx, step=1.0)
cy = st.sidebar.slider("centre y (m)", float(np.floor(bmin[1])), float(np.ceil(bmax[1])), key=ky, step=1.0)
size = st.sidebar.slider("window size (m)", 5, 250, key=ks, step=5)
max_pts = st.sidebar.slider("max points in 3-D", 10_000, 250_000, 60_000, step=10_000,
                            help="Higher = more detail but more browser/RAM load")
psize = st.sidebar.slider("point size", 1, 6, 2)

half = size / 2
xyz = d["xyz"]
win_mask = (np.abs(xyz[:, 0] - cx) <= half) & (np.abs(xyz[:, 1] - cy) <= half)
win_idx = np.flatnonzero(win_mask)
view_idx = subsample(win_idx, max_pts)
rect = (cx - half, cx + half, cy - half, cy + half)
st.sidebar.caption(f"{len(win_idx):,} points in window, showing {len(view_idx):,}")
if len(win_idx) == 0:
    st.sidebar.warning("The window is empty - box-select on the top-down map to place it on the street.")

sources = ["RGB", "Reference labels", "Intensity", "Height (z)"]
for r in runs:
    sources += [f"{r} · prediction", f"{r} · errors", f"{r} · confidence"]


def handle_box(event, key):
    try:
        boxes = event.selection.box if event else []
    except AttributeError:
        boxes = []
    if boxes:
        b = boxes[0]
        x0, x1 = sorted(b["x"])
        y0, y1 = sorted(b["y"])
        sig = (wid, round(x0, 1), round(x1, 1), round(y0, 1), round(y1, 1))
        if st.session_state.get(f"last_box_{key}") != sig:
            st.session_state[f"last_box_{key}"] = sig
            s = int(min(250, max(5, 5 * round(max(x1 - x0, y1 - y0) / 5))))
            st.session_state["pending_window"] = (wid, float(round((x0 + x1) / 2)), float(round((y0 + y1) / 2)), s)
            st.rerun()


# ----------------------------------------------------------------------------- tabs
st.title(f"{cfg['data'].get('name', 'Toronto-3D')} · {tile}")
if cfg["data"].get("synthetic"):
    st.warning("**Synthetic test data** - a generated street (same file format, tile names and coordinate "
               "frame as Toronto-3D) used only to check the pipeline works. Scores here are meaningless; "
               "switch *Dataset config* to `default.yaml` for the real Toronto-3D scans.", icon="🧪")
# One page at a time (st.tabs would build every tab's figures on each rerun).
PAGES = ["🔍 Explore", "⚖️ Compare", "📊 Metrics", "📍 Inventory", "🧪 Experiments"]
page = st.segmented_control("Page", PAGES, default=PAGES[0], key="page", label_visibility="collapsed") or PAGES[0]

ov = overview_index(len(xyz))

if page == "🔍 Explore":
    c1, c2 = st.columns([3, 1])
    with c1:
        ov_src = st.radio("Top-down colour", ["Reference labels"] + [f"{r} · prediction" for r in runs],
                          horizontal=True, key="ov_src")
        lab = d["label"][ov] if ov_src == "Reference labels" else preds[ov_src.split(" · ")[0]]["pred"][ov]
        fig = fig_topdown(xyz[ov, :2], lambda: label_traces(xyz[ov, 0], xyz[ov, 1], None, lab, 2, three_d=False), rect)
        handle_box(st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="box",
                                   key="ov_explore"), "explore")
    with c2:
        st.markdown("**Reference classes in window**")
        st.caption("Ground-truth labels inside the black square (unclassified is not scored).")
        cnt = np.bincount(d["label"][win_idx], minlength=9)
        dist = pd.DataFrame({"class": [CLASS_NAMES[c] for c in range(9)], "points": cnt,
                             "share %": (100 * cnt / max(cnt.sum(), 1)).round(1)})
        st.dataframe(dist[dist.points > 0], hide_index=True, width="stretch")
    src = st.selectbox("3-D colour", sources, index=1, key="explore_src")
    kind, vals, title = colour_spec(src, view_idx, d, preds)
    st.plotly_chart(fig_3d(xyz[view_idx], kind, vals, psize, title), width="stretch")
    st.caption("Drag to rotate · scroll to zoom · click legend entries to hide/show classes")

if page == "⚖️ Compare":
    if not runs:
        st.info("No predictions for this tile yet - run `python -m streetinv.baseline` or `python -m streetinv.train`.")
    a, b = st.columns(2)
    left = a.selectbox("Left", sources, index=1, key="cmp_l")
    right = b.selectbox("Right", sources, index=sources.index(f"{runs[0]} · prediction") if runs else 0, key="cmp_r")
    for col, s, side in ((a, left, "l"), (b, right, "r")):
        kind, vals, title = colour_spec(s, view_idx, d, preds)
        col.plotly_chart(fig_3d(xyz[view_idx], kind, vals, psize, title, height=540, rev="cmp"),
                         width="stretch", key=f"cmp3d_{side}")

    if runs:
        st.subheader("Scores inside the window")
        rows = []
        for r in runs:
            m, _ = evaluate(d["label"][win_idx], preds[r]["pred"][win_idx])
            rows.append({"run": r, **{k.replace("IoU_", ""): round(v, 1) for k, v in m.items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if len(runs) >= 2:
        st.subheader("Where do two models disagree?")
        c1, c2 = st.columns(2)
        ra = c1.selectbox("Model A", runs, index=0, key="dis_a")
        rb = c2.selectbox("Model B", runs, index=1, key="dis_b")
        gt = d["label"][ov]
        okA, okB = preds[ra]["pred"][ov] == gt, preds[rb]["pred"][ov] == gt
        cat = np.select([gt == 0, okA & okB, okA & ~okB, ~okA & okB], [4, 0, 1, 2], 3)
        spec = [(0, "both correct", OK_GREY), (1, f"only {ra} correct", run_color[ra]),
                (2, f"only {rb} correct", run_color[rb]), (3, "both wrong", ERR_RED)]

        def dis_traces():
            return [go.Scattergl(x=xyz[ov][cat == c, 0], y=xyz[ov][cat == c, 1], mode="markers", name=n,
                                 marker=dict(size=2 if c == 0 else 3, color=col)) for c, n, col in spec if (cat == c).any()]

        handle_box(st.plotly_chart(fig_topdown(None, dis_traces, rect), width="stretch", on_select="rerun",
                                   selection_mode="box", key="ov_dis"), "dis")
        valid = gt > 0
        st.caption(" · ".join(f"{n}: {100 * (cat[valid] == c).mean():.1f}%" for c, n, _ in spec))

if page == "📊 Metrics":
    rows = []
    for r in runs:
        mf = pdir / r / f"{tile}_metrics.json"
        if mf.exists():
            j = json.loads(mf.read_text())
            rows.append({"model": r, "family": j.get("family", ""),
                         **{k.replace("IoU_", ""): v for k, v in j["metrics"].items()}, "_cm": j["confusion"]})
    if not rows:
        st.info("No evaluated predictions for this tile.")
    else:
        df = pd.DataFrame(rows)
        show = df.drop(columns="_cm")
        if is_test and cfg_name == "default.yaml":
            ref = pd.DataFrame([{"model": k, "family": "published", **v} for k, v in REFERENCE_RESULTS.items()])
            show = pd.concat([show, ref], ignore_index=True)
        st.dataframe(show.style.format(precision=1), hide_index=True, width="stretch")
        if not is_test:
            st.warning("This is a training tile - these scores are not a fair test.")

        cls = [CLASS_NAMES[c] for c in EVAL_CLASSES]
        long = show.melt(id_vars=["model"], value_vars=["mIoU"] + cls, var_name="class", value_name="IoU")
        cmap = {**run_color, **{k: REF_GREY for k in REFERENCE_RESULTS}}
        fig = px.bar(long, x="class", y="IoU", color="model", barmode="group", color_discrete_map=cmap,
                     height=380)
        fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>IoU %{y:.1f}%<extra>%{fullData.name}</extra>")
        fig.update_layout(bargap=0.25, bargroupgap=0.08, yaxis=dict(title="IoU (%)", range=[0, 100], gridcolor="#ecebe6"),
                          xaxis_title=None, margin=dict(t=10, b=0), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")

        sel = st.selectbox("Confusion matrix of", df.model.tolist())
        cm = np.array(df.set_index("model").loc[sel, "_cm"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig = px.imshow(100 * norm, x=cls, y=cls, text_auto=".0f", color_continuous_scale=SEQ, zmin=0, zmax=100,
                        labels=dict(x="predicted", y="reference", color="% of row"), aspect="auto", height=460)
        fig.update_layout(margin=dict(t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Rows = reference class, columns = predicted class, values = % of the reference class.")

if page == "📍 Inventory":
    if not runs:
        st.info("Run a model first.")
    else:
        from streetinv.inventory import build_inventory

        r = st.selectbox("Inventory from model", runs, key="inv_run")
        idir = inventory_dir(cfg) / r
        objf, reff, sumf = idir / f"{tile}_objects.csv", idir / f"{tile}_reference.csv", idir / f"{tile}_summary.json"
        stale = not objf.exists() or mtime(objf) < mtime(pdir / r / f"{tile}.npz")
        if stale:
            st.info("No (up-to-date) inventory for this run.")
            if st.button("Build inventory", type="primary"):
                with st.spinner("Extracting objects…"):
                    build_inventory(cfg, r, tile, d)
                st.rerun()
        else:
            summ = json.loads(sumf.read_text())
            objs, refs = pd.read_csv(objf), pd.read_csv(reff)
            for name, s in summ.items():
                st.markdown(f"**{name.capitalize()}s** - {s['n_pred']} detected vs {s['n_ref']} in reference")
                cols = st.columns(5)
                cols[0].metric("Precision", f"{s['precision']:.2f}" if s["precision"] == s["precision"] else "-")
                cols[1].metric("Recall", f"{s['recall']:.2f}" if s["recall"] == s["recall"] else "-")
                cols[2].metric("F1", f"{s['F1']:.2f}")
                cols[3].metric("Mean position error", f"{s['mean_pos_error_m']:.2f} m" if s["TP"] else "-")
                cols[4].metric("Mean |height error|", f"{s['mean_abs_height_error_m']:.2f} m" if s["TP"] else "-")
            st.caption("Reference objects are built from the ground-truth labels with the same procedure "
                       "(Toronto-3D has no instance labels). TP = matched within "
                       f"{cfg['inventory']['pole']['match_dist']} m (poles) / {cfg['inventory']['car']['match_dist']} m (cars).")

            present = sorted(set(objs["class"]) | set(refs["class"]))
            show_cls = st.multiselect("Classes", present, default=["pole"] if "pole" in present else present)
            mp = pd.concat([objs.assign(source="detected"),
                            refs[refs.status == "FN"].assign(source="missed (reference)")], ignore_index=True)
            mp = mp[mp["class"].isin(show_cls)]
            c1, c2 = st.columns([3, 2])
            with c1:
                view = st.radio("Map", ["Web map", "Local x/y"], horizontal=True, key="inv_map")
                if len(mp):
                    hover = {"obj_id": True, "class": True, "height": ":.1f", "n_points": True, "source": True}
                    if view == "Web map":
                        fig = px.scatter_map(mp, lat="lat", lon="lon", color="status", color_discrete_map=STATUS,
                                             hover_data=hover, zoom=17, height=480, map_style="carto-positron")
                        fig.update_traces(marker=dict(size=11))
                    else:
                        fig = px.scatter(mp, x="x", y="y", color="status", color_discrete_map=STATUS,
                                         hover_data=hover, height=480)
                        fig.update_traces(marker=dict(size=10, line=dict(width=2, color="white")))
                        fig.update_yaxes(scaleanchor="x")
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.05))
                    st.plotly_chart(fig, width="stretch")
                st.caption("🟢 TP correct detection · 🔴 FP false alarm · 🟠 FN missed reference object")
            with c2:
                cols = ["obj_id", "class", "status", "source", "height", "n_points", "match_dist", "x", "y", "lat", "lon"]
                table = mp[[c for c in cols if c in mp.columns]].round(2).reset_index(drop=True)
                ev = st.dataframe(table, hide_index=True, width="stretch", height=440,
                                  on_select="rerun", selection_mode="single-row", key="inv_table")
                c3, c4 = st.columns(2)
                c3.download_button("⬇ objects.csv", objf.read_bytes(), f"{r}_{tile}_objects.csv")
                c4.download_button("⬇ objects.geojson", (idir / f"{tile}_objects.geojson").read_bytes(),
                                   f"{r}_{tile}_objects.geojson")
            rows_sel = ev.selection.rows if ev else []
            if rows_sel:
                o = table.iloc[rows_sel[0]]
                st.subheader(f"{o['class']} #{o['obj_id']} ({o['status']}, {o['source']})")
                rad = st.slider("crop radius (m)", 2, 20, 6)
                m = np.flatnonzero(np.hypot(xyz[:, 0] - o.x, xyz[:, 1] - o.y) <= rad)
                m = subsample(m, 80_000)
                a, b = st.columns(2)
                k1, v1, t1 = colour_spec("Reference labels", m, d, preds)
                a.plotly_chart(fig_3d(xyz[m], k1, v1, psize + 1, t1, 520, "obj"), width="stretch", key="obj_l")
                k2, v2, t2 = colour_spec(f"{r} · prediction", m, d, preds)
                b.plotly_chart(fig_3d(xyz[m], k2, v2, psize + 1, t2, 520, "obj"), width="stretch", key="obj_r")
                if st.button("Move 3-D window here"):
                    st.session_state["pending_window"] = (wid, float(round(o.x)), float(round(o.y)), 20)
                    st.rerun()
            else:
                st.caption("Select a row in the table to inspect the object in 3-D.")

if page == "🧪 Experiments":
    try:
        from streetinv.tracking import setup_mlflow

        mlflow = setup_mlflow(cfg)
        exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment"])
        runs_df = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"]) if exp else pd.DataFrame()
    except Exception as e:  # noqa: BLE001
        runs_df = pd.DataFrame()
        st.error(f"Could not read MLflow: {e}")
    if runs_df.empty:
        st.info("No MLflow runs yet.")
    else:
        keep = ["tags.mlflow.runName", "tags.family", "tags.model", "status", "start_time"]
        mcols = [c for c in runs_df.columns if c.startswith("metrics.") and
                 any(s in c for s in ("_mIoU", "_OA", "inv_pole_F1", "inv_car_F1", "train_seconds"))]
        tab = runs_df[["run_id"] + [c for c in keep if c in runs_df] + sorted(mcols)].copy()
        tab.columns = [c.replace("metrics.", "").replace("tags.mlflow.runName", "run").replace("tags.", "") for c in tab.columns]
        st.dataframe(tab.round(2), hide_index=True, width="stretch")
        st.caption("Full UI: `mlflow ui --backend-store-uri " + cfg["mlflow"]["tracking_uri"] + "`")
        trainable = tab[tab["family"] == "deep learning"] if "family" in tab else tab
        opts = {f"{r.run} · {r.run_id[:8]}": r.run_id for r in trainable.itertuples()}
        chosen = st.multiselect("Training curves", list(opts), default=list(opts)[:2])
        if chosen:
            client = mlflow.tracking.MlflowClient()
            c1, c2 = st.columns(2)
            for col, metric, label in ((c1, "val_mIoU", "validation mIoU (%)"), (c2, "train_loss", "training loss")):
                rows = [{"run": k, "epoch": h.step + 1, "value": h.value}
                        for k in chosen for h in client.get_metric_history(opts[k], metric)]
                if rows:
                    fig = px.line(pd.DataFrame(rows), x="epoch", y="value", color="run", height=320,
                                  color_discrete_sequence=RUN_COLORS, markers=True)
                    fig.update_traces(line_width=2, marker_size=8)
                    fig.update_layout(yaxis_title=label, margin=dict(t=10, b=0), legend=dict(orientation="h", y=1.15),
                                      hovermode="x unified", yaxis_gridcolor="#ecebe6")
                    col.plotly_chart(fig, width="stretch")
            st.caption("Validation = fixed random crops of the test tile, used only for monitoring (not model selection).")
