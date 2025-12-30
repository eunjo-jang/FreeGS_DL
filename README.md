# FreeGS_DL — Synthetic Psi Reconstruction (Baseline MLP)

⚠️ Work-in-progress baseline; outputs are rough and need improvement.

## What this project does
- Use FreeGS (Grad–Shafranov solver) to generate synthetic tokamak equilibria and sensor readings (flux loops, magnetic probes, Rogowski).
- Train a simple MLP to map 1D sensor features to a 2D normalized psi grid.
- Evaluate and visualize ground truth vs prediction contours.

## Layout
```
FreeGS_DL/
├─ configs/defaults.yaml      # config for paths, seeds, hparams
├─ data/dataset_freegs/       # generated data (X.npy, Y_psi.npy, meta.json, sensors.json)
├─ data/splits.json           # generated split indices (ignored by git; regenerate)
├─ assets/image/              # saved plots (GT vs Pred)
├─ artifacts/mlp_best.pt      # best checkpoint (x_mean/x_std included)
├─ src/
│   ├─ data_gen.py            # data generation (FreeGS → X/Y/meta/sensors)
│   ├─ splits.py              # make train/val/test splits
│   ├─ dataset.py             # torch Dataset + loading helpers
│   ├─ model.py               # MLP definition
│   ├─ train.py               # training loop + early stopping
│   └─ eval.py                # evaluation + plotting
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

## Pipeline (new entrypoints)
- Generate data (defaults: 200 samples, 65×65 grid):
  ```
  python -m src.data_gen --config configs/defaults.yaml
  # outputs to data/dataset_freegs/{X.npy,Y_psi.npy,meta.json,sensors.json}
  ```
- Make splits (seed from config):
  ```
  python -m src.splits --config configs/defaults.yaml
  # writes data/splits.json (ignored by git)
  ```
- Train baseline MLP:
  ```
  python -m src.train --config configs/defaults.yaml
  # saves best to artifacts/mlp_best.pt (with x_mean/x_std)
  ```
- Evaluate & save plots:
  ```
  python -m src.eval --config configs/defaults.yaml --num-examples 4
  # saves images to assets/image/
  ```

## Notes & gaps
- Model is a simple fully-connected MLP; predictions are noisy. Consider CNN/U-Net or better conditioning for improved spatial fidelity.
- Randomness: split seed is fixed; training loop has no deterministic mode beyond seeding, so results may vary slightly.
- SciPy is optional; without it, interpolation falls back to nearest-neighbor.

## Current included data/artifacts
- `data/dataset_freegs/` currently contains X.npy (121, 41) and Y_psi.npy (121, 65, 65) from an earlier run.
- `artifacts/mlp_best.pt` is a checkpoint from that run.
- `assets/image/` has the corresponding GT vs Pred contour PNGs.

