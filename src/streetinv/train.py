"""Train RandLA-Net on Toronto-3D, then predict the test tile(s).

    python -m streetinv.train [--run-name randla] [--epochs 40] [--num-points 32768]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate

from .config import grid_dir, load_config, resolve
from .dataset import (D_IN, FEATURE_NAMES, CropDataset, available_ram_mb, batch_to_device, class_weights,
                      load_tiles, make_crop, to_torch)
from .infer import run_inference
from .metrics import confusion, mlflow_safe, scores
from .randla import RandLANet
from .tracking import setup_mlflow


def fixed_val_batches(tiles, rc, n_crops, seed=123):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_crops):
        t = tiles[i % len(tiles)]
        s, _, _ = make_crop(t, int(rng.integers(len(t.xyz))), rc["num_points"], rc["k"], rng)
        samples.append(to_torch(s))
    bs = rc["batch_size"]
    return [default_collate(samples[i:i + bs]) for i in range(0, len(samples), bs)]


@torch.no_grad()
def validate(model, batches, device, amp):
    model.eval()
    cm = np.zeros((8, 8), np.int64)
    for b in batches:
        b = batch_to_device(b, device)
        with torch.autocast(device.type, enabled=amp):
            pred = model(b).argmax(-1)
        cm += confusion(b["labels"].cpu().numpy().ravel() + 1, pred.cpu().numpy().ravel() + 1)
    return scores(cm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-name", default="randla")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--steps-per-epoch", type=int)
    ap.add_argument("--num-points", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-test", action="store_true", help="skip the full-tile test inference")
    a = ap.parse_args(argv)
    ov = {k: getattr(a, k.replace("-", "_")) for k in
          ("epochs", "steps_per_epoch", "num_points", "batch_size", "num_workers")}
    cfg = load_config(a.config, {"randla": {k: v for k, v in ov.items() if v is not None}})
    rc = cfg["randla"]
    assert rc["num_points"] % 256 == 0, "num_points must be divisible by 256"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    # NB: do not touch CUDA before the DataLoader workers are forked - otherwise every
    # worker inherits the parent's CUDA host memory (> 1 GB) and 8 GB machines run out of RAM.
    use_cuda = torch.cuda.is_available()

    gdir = grid_dir(cfg)
    train_tiles = load_tiles(gdir, cfg["data"]["train_tiles"])
    weights = class_weights(train_tiles)
    ds = CropDataset(train_tiles, rc["num_points"], rc["k"], rc["steps_per_epoch"] * rc["batch_size"])
    loader = DataLoader(ds, batch_size=rc["batch_size"], num_workers=rc["num_workers"],
                        pin_memory=use_cuda, persistent_workers=rc["num_workers"] > 0, drop_last=True)
    if rc["num_workers"] > 0:
        iter(loader)  # fork the persistent workers now, while the process is still small
    test_tiles = load_tiles(gdir, cfg["data"]["test_tiles"])
    val_batches = fixed_val_batches(test_tiles, rc, rc["val_crops"])
    del test_tiles  # reloaded after training, keeps peak RAM down
    print(f"RAM available after loading data: {available_ram_mb():.0f} MB")
    if available_ram_mb() < 1500:
        print("WARNING: low RAM - consider --num-workers 0/1, a larger grid_size, or more WSL memory")

    device = torch.device("cuda" if use_cuda else "cpu")
    amp = bool(rc["amp"]) and use_cuda

    model = RandLANet(D_IN, 8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rc["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=rc["lr_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device), ignore_index=-1)
    ckpt_path = resolve("models") / f"{a.run_name}.pt"

    mlflow = setup_mlflow(cfg)
    with mlflow.start_run(run_name=a.run_name) as run:
        mlflow.set_tags({"family": "deep learning", "model": "RandLA-Net", "features": ",".join(FEATURE_NAMES),
                         "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"})
        mlflow.log_params({**{f"randla.{k}": v for k, v in rc.items()},
                           "grid_size": cfg["data"]["grid_size"],
                           "train_tiles": ",".join(cfg["data"]["train_tiles"]),
                           "test_tiles": ",".join(cfg["data"]["test_tiles"]),
                           "n_params": sum(p.numel() for p in model.parameters())})
        mlflow.log_dict({"class_weights": weights.tolist()}, "class_weights.json")
        t_start = time.time()
        for epoch in range(rc["epochs"]):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for b in loader:
                b = batch_to_device(b, device)
                with torch.autocast(device.type, enabled=amp):
                    logits = model(b)
                loss = crit(logits.float().reshape(-1, 8), b["labels"].reshape(-1))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
                n += 1
            sched.step()
            vm = validate(model, val_batches, device, amp)
            mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
            mlflow.log_metrics({"train_loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"],
                                "epoch_seconds": time.time() - t0, "gpu_mem_mb": mem,
                                **mlflow_safe(vm, "val_")}, step=epoch)
            print(f"epoch {epoch + 1:3d}/{rc['epochs']}  loss {tot / max(n, 1):.4f}  "
                  f"val OA {vm['OA']:.2f} mIoU {vm['mIoU']:.2f}  ({time.time() - t0:.0f}s, "
                  f"GPU {mem:.0f} MB, free RAM {available_ram_mb():.0f} MB)", flush=True)
            torch.save({"model": model.state_dict(), "cfg": cfg, "d_in": D_IN, "epoch": epoch + 1,
                        "run_name": a.run_name}, ckpt_path)
        mlflow.log_metric("train_seconds", time.time() - t_start)
        mlflow.log_artifact(str(ckpt_path), "model")
        del loader, ds, train_tiles
        if not a.no_test:
            test_tiles = load_tiles(gdir, cfg["data"]["test_tiles"])
            run_inference(cfg, model, test_tiles, a.run_name, run.info.run_id, mlflow, device)


if __name__ == "__main__":
    main()
