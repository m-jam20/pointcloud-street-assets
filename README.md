# Street furniture inventory from mobile laser scanning (Toronto-3D)

Turn a mobile LiDAR scan of a street into a **map-ready inventory of street furniture**
(poles - lamp posts, traffic lights, sign posts, utility poles - and parked cars),
and compare a **conventional** point-cloud pipeline against a **deep-learning** one.

| | |
|---|---|
| **Input** | MLS point cloud (XYZ, RGB, intensity), e.g. a car-mounted Teledyne Optech Maverick scan |
| **Output** | per-point classes (road, road marking, natural, building, utility line, pole, car, fence) **and** an object inventory (CSV / GeoJSON: position, lat/lon, height, size) |
| **Who uses it** | municipalities, utilities and mapping companies maintaining asset registers / digital twins |
| **Added value** | replaces manual digitising of assets in point-cloud viewers; repeatable, measurable accuracy |

## Two approaches, same evaluation

| | Conventional | Deep learning |
|---|---|---|
| Method | handcrafted geometry (eigen-features at 2 scales, verticality, height above ground, colour, intensity) + Random Forest | **RandLA-Net** (Hu et al. 2020), pure PyTorch, trained on random 32k-point crops |
| Code | [src/streetinv/baseline.py](src/streetinv/baseline.py) | [src/streetinv/randla.py](src/streetinv/randla.py), [train.py](src/streetinv/train.py) |
| Typical mIoU on Toronto-3D (literature) | ~ 50-65 % | 77-84 % (RandLA-Net paper: 81.8 % with RGB) |

Both feed the same **inventory step** ([inventory.py](src/streetinv/inventory.py)): pole / car points
-> 2-D connected components -> objects with position, height and footprint, matched (Hungarian,
≤ 1 m for poles, ≤ 2 m for cars) against objects built the same way from the ground-truth labels.
Toronto-3D has no instance labels, so this reference is derived, not hand-made.

All runs are tracked in **MLflow** (params, per-class IoU, confusion matrices, inventory P/R/F1, model files).

## Results (test tile L002)

_Fill in after training on the real data — `python -m streetinv.report` prints this table._

## Quick start

```bash
# 0. environment (Python 3.10, CUDA GPU optional; tested on a GTX 1650 4 GB + 8 GB RAM, WSL2)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"

# 1. data - OneDrive does not allow scripted downloads: get Toronto_3D.zip in a browser from
#    https://1drv.ms/u/s!Amlc6yZnF87psX6hKS8VOQllVvj4?e=yWhrYX  then
python scripts/download_toronto3d.py --zip /mnt/c/Users/<you>/Downloads/Toronto_3D.zip

# 2. voxel-subsample (0.10 m) -> data/processed/grid_0.10/
python -m streetinv.prepare

# 3. conventional baseline  (~10 min, CPU)
python -m streetinv.baseline

# 4. RandLA-Net  (GPU; ~1-2 h with defaults)  -> models/randla.pt + test predictions
python -m streetinv.train

# 5. inventories
python -m streetinv.inventory --run rf_baseline
python -m streetinv.inventory --run randla

# 6. look at everything
streamlit run app/streamlit_app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db      # optional, full MLflow UI
```

No data yet? Run the whole pipeline on a synthetic street in the same PLY format:

```bash
python scripts/make_synthetic.py
python -m streetinv.prepare  --config configs/synthetic.yaml
python -m streetinv.baseline --config configs/synthetic.yaml
python -m streetinv.train    --config configs/synthetic.yaml
python -m streetinv.inventory --config configs/synthetic.yaml --run randla
```

## The Streamlit app

Pick the dataset config and tile in the sidebar; box-select on any top-down map to move the 3-D window.

* **Explore** - top-down overview + 3-D view coloured by RGB, reference labels, predictions, errors,
  confidence, intensity or height; click legend entries to hide classes.
* **Compare** - two synchronised-window 3-D views side by side (e.g. reference vs RandLA-Net, or
  baseline errors vs RandLA-Net errors), window-level scores, and a map of where two models disagree.
* **Metrics** - OA / mIoU / per-class IoU for every run next to published results, confusion matrices.
* **Inventory** - precision / recall / F1, position and height errors, web map of TP / FP / missed
  objects, object table with CSV / GeoJSON download, and a 3-D close-up of any selected object.
* **Experiments** - MLflow runs and training curves.

## Design notes

* **Coordinates**: UTM 17N (NAD83, EPSG:26917) with the authors' offset `[627285, 4841948, 0]`
  subtracted, so float32 keeps millimetre precision.
* **Split**: official - train L001, L003, L004; test L002. Validation crops during training come from
  L002 and are used for monitoring only; the last checkpoint is evaluated (no model selection on test).
* **Low-RAM friendly** (8 GB): PLY tiles are memory-mapped and voxelised chunk by chunk; DataLoader
  workers are forked before CUDA is initialised; the app renders one page at a time.
  If you still hit out-of-memory, use `--num-workers 0`, a larger `grid_size`, or give WSL more memory
  in `%UserProfile%\.wslconfig` (`[wsl2]` / `memory=10GB` / `swap=8GB`).
* **Licence**: Toronto-3D is CC BY-NC 4.0 - data, derived tiles and predictions are never committed
  (see `.gitignore`); only code and the download script are. Paris-Lille-3D (CC BY-NC-ND) can be added
  later the same way.

## Layout

```
configs/            default.yaml (Toronto-3D), synthetic.yaml (smoke test)
scripts/            download_toronto3d.py, make_synthetic.py
src/streetinv/      io, prepare, features, baseline, randla, dataset, train, infer, inventory, metrics, tracking
app/                streamlit_app.py
tests/              unit tests (pytest)
```

## References

* Tan et al. (2020) *Toronto-3D: A large-scale mobile LiDAR dataset for semantic segmentation of urban roadways.* CVPRW.
* Hu et al. (2020) *RandLA-Net: Efficient semantic segmentation of large-scale point clouds.* CVPR.
* Weinmann et al. (2015) *Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers.* ISPRS J.
