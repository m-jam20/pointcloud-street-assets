"""Full-tile inference with RandLA-Net ("possibility" sweep as in the original paper).

    python -m streetinv.infer --checkpoint models/randla.pt [--tiles L002] [--run-name randla]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import default_collate
from tqdm import tqdm

from .config import CLASS_NAMES, grid_dir, load_config, resolve
from .dataset import D_IN, batch_to_device, load_tiles, make_crop, to_torch
from .metrics import evaluate, mlflow_safe
from .randla import RandLANet
from .tracking import confusion_figure, save_prediction, setup_mlflow


@torch.no_grad()
def infer_tile(model, tile, rc: dict, device, seed=0, verbose=True):
    model.eval()
    rng = np.random.default_rng(seed)
    n = len(tile.xyz)
    probs = np.zeros((n, 8), np.float32)
    poss = rng.random(n) * 1e-3
    thr = rc["infer_threshold"]
    bar = tqdm(total=100, desc=f"infer {tile.name}", disable=not verbose, unit="%")
    done = 0
    while True:
        m = poss.min()
        prog = int(100 * min(m / thr, 1))
        bar.update(prog - done)
        done = prog
        if m >= thr:
            break
        samples, idxs = [], []
        for _ in range(rc["batch_size"]):
            c = int(np.argmin(poss))
            s, idx, d2 = make_crop(tile, c, rc["num_points"], rc["k"], rng, augment=False)
            poss[idx] += (1 - d2 / max(d2.max(), 1e-9)) ** 2 + 1e-3
            s.pop("labels")
            samples.append(to_torch(s))
            idxs.append(idx)
        b = batch_to_device(default_collate(samples), device)
        with torch.autocast(device.type, enabled=rc.get("amp", True) and device.type == "cuda"):
            p = torch.softmax(model(b).float(), -1).cpu().numpy()
        for i, idx in enumerate(idxs):
            probs[idx] += p[i]
    bar.close()
    pred = probs.argmax(1).astype(np.uint8) + 1
    conf = probs.max(1) / np.maximum(probs.sum(1), 1e-9)
    return pred, conf


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = RandLANet(ck.get("d_in", D_IN), 8).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def run_inference(cfg, model, tiles, run_name, run_id=None, mlflow=None, device=None, seed=0):
    rc = cfg["randla"]
    results = {}
    for tile in tiles:
        t0 = time.time()
        pred, conf = infer_tile(model, tile, rc, device, seed)
        m, cm = evaluate(tile.label, pred)
        results[tile.name] = m
        save_prediction(cfg, run_name, tile.name, pred, conf, m, cm,
                        {"mlflow_run_id": run_id, "family": "deep learning"})
        if mlflow is not None:
            mlflow.log_metric(f"{tile.name}_infer_seconds", time.time() - t0)
            mlflow.log_metrics(mlflow_safe(m, f"{tile.name}_"))
            mlflow.log_figure(confusion_figure(cm, f"{run_name} - {tile.name}"), f"confusion_{tile.name}.png")
        print(f"{tile.name}: OA {m['OA']:.2f}  mIoU {m['mIoU']:.2f}")
        for c in range(1, 9):
            print(f"   {CLASS_NAMES[c]:>13s}: {m['IoU_' + CLASS_NAMES[c]]:.2f}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tiles", nargs="+", default=None)
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(resolve(a.checkpoint), device)
    cfg = load_config(a.config)
    cfg["randla"].update({k: ck["cfg"]["randla"][k] for k in ("num_points", "k")})
    names = a.tiles or cfg["data"]["test_tiles"]
    tiles = load_tiles(grid_dir(cfg), names)
    run_name = a.run_name or ck.get("run_name", "randla")
    mlflow = setup_mlflow(cfg)
    with mlflow.start_run(run_name=f"{run_name}_infer") as run:
        mlflow.set_tags({"family": "deep learning", "model": "RandLA-Net", "stage": "inference"})
        mlflow.log_param("checkpoint", str(a.checkpoint))
        run_inference(cfg, model, tiles, run_name, run.info.run_id, mlflow, device)


if __name__ == "__main__":
    main()
