# FreeGS_DL — Synthetic Psi Reconstruction (Baseline MLP)

⚠️ Work-in-progress baseline; outputs are rough and need improvement.

## What this project does
- Use FreeGS (Grad–Shafranov solver) to generate synthetic tokamak equilibria and sensor readings (flux loops, magnetic probes, Rogowski).
- Train a simple MLP to map 1D sensor features to a 2D normalized psi grid.
- Evaluate and visualize ground truth vs prediction contours.

## Layout
```
FreeGS_DL/
├─ configs/                   # per-model configs (mlp.yaml, coord_mlp.yaml, ...)
├─ data/dataset_freegs/       # generated data (X.npy, Y_psi.npy, meta.json, sensors.json)
├─ data/splits.json           # generated split indices (ignored by git; regenerate)
├─ figures/                   # saved plots per model (GT vs Pred)
├─ checkpoints/               # checkpoints (mlp_best.pt, coord_mlp_best.pt, ...)
├─ src/
│   ├─ data_gen.py            # data generation (FreeGS → X/Y/meta/sensors)
│   ├─ splits.py              # make train/val/test splits
│   ├─ dataset.py             # torch Dataset + loading helpers
│   ├─ models/                # model zoo
│   │    ├─ mlp.py            # baseline MLP
│   │    ├─ coord_mlp.py      # coord-aug MLP (placeholder)
│   │    ├─ deeponet.py       # DeepONet-style (placeholder)
│   │    └─ pinn.py            # PINN-style (placeholder)
│   │    
│   ├─ train.py               # training loop (model selection via config)
│   └─ eval.py                # evaluation + plotting (supports coord_mlp pointwise eval)
├─ scripts/                   # optional helper shell scripts (empty by default)
└─ README.md
```

## Requirements
- Python 3.10+ (tested: Python 3.13.5 on macOS, MPS; CUDA auto-detected if available).
- PyTorch (install matching your platform; see [pytorch.org](https://pytorch.org/get-started/locally/)).
- FreeGS (install separately; see below).
- NumPy, SciPy (optional but recommended for smoother interpolation), Matplotlib, PyYAML.

## Quickstart
```bash
git clone <your-repo-url>
cd freegs/FreeGS_DL

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Install PyTorch (choose command for your platform, e.g. CPU-only):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install FreeGS
# Option A) pip (if available):
# pip install freegs
# Option B) from source (recommended):
git clone https://github.com/fsmetana/freegs.git
pip install -e freegs

# Project deps
pip install numpy scipy matplotlib pyyaml
```

## Pipeline (choose config per model)
- Generate data (defaults: 200 samples, 65×65 grid):
  ```
  python -m src.data_gen --config configs/mlp.yaml
  # outputs to data/dataset_freegs/{X.npy,Y_psi.npy,meta.json,sensors.json}
  ```
- Make splits (seed from config):
  ```
  python -m src.splits --config configs/mlp.yaml
  # writes data/splits.json (ignored by git)
  ```
- Train model (select via config `model.name`: mlp, coord_mlp, deeponet, pinn, cnn):
  ```
  python -m src.train --config configs/mlp.yaml
# saves best to checkpoints/<model>_best.pt (with x_mean/x_std)
  ```
- Evaluate & save plots:
  ```
  python -m src.eval --config configs/mlp.yaml --num-examples 4
# saves images to figures/<model>/
  ```

## Current included data/checkpoints/figures
- `data/dataset_freegs/` currently contains X.npy (121, 41) and Y_psi.npy (121, 65, 65) from an earlier run.
- `checkpoints/*.pt` are the saved model weights (per model).
- `figures/<model>/` holds generated plots per model; `figures/gt/` keeps the reference plots.

## Quick usage (train & eval)
```bash
cd freegs/FreeGS_DL

# (once) create env & install deps
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # choose CUDA wheel on GPU
pip install -r requirements.txt

# train
python -m src.train --config configs/mlp.yaml
# -> saves to checkpoints/mlp_best.pt

# eval & plot
python -m src.eval --config configs/mlp.yaml --num-examples 4
# -> saves plots to figures/mlp/
```

## Experiment results 
| Model      | MSE  | RelErr  | Spatial MSE  |
|------------|------------|---------------|--------------------|
| mlp        | 0.003460        | 0.062236          | 0.003460                |
| coord_mlp  | **0.000238**       | 0.024426         | **0.000238**             |
| deeponet   | 0.000324        | 0.027931          | 0.000324               |
| pinn       |0.000243        | **0.023670**          | 0.000243                |

> Run `python -m src.eval --config configs/<model>.yaml` to populate the metrics and plots.

