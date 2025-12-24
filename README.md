# FreeGS_DL — Synthetic Psi Reconstruction (Baseline MLP)

⚠️ Work-in-progress baseline; outputs are rough and need improvement.

## What this project does
- Use FreeGS (a Grad–Shafranov solver) to generate synthetic magnetic equilibria and sensor readings (flux loops, magnetic probes, Rogowski).
- Train a simple MLP to map 1D sensor features to a 2D normalized psi grid.
- Evaluate and visualize ground truth vs prediction contours (see `image/`).

## Repo Layout (FreeGS_DL/)
```
FreeGS_DL/
├─ generate_dataset_freegs.py     # synth equilibria + sensor sampling → X.npy, Y_psi.npy, meta.json, sensors.json
├─ make_splits.py                 # splits.json (train/val/test indices, seed=42)
├─ train_baseline_mlp.py          # MLP training, saves mlp_best.pt (includes x_mean/x_std)
├─ eval_and_plot.py               # eval + contour plots (GT vs Pred)
├─ inspect_dataset.py             # quick dataset inspection
├─ check_case_ab.py               # sample-case sanity checks
├─ dataset_freegs/                # included dataset (current: X.npy (121, 41), Y_psi.npy (121, 65, 65))
├─ image/                         # eval_plot_Figure_1~4_(True|Pred).png
├─ splits.json                    # train/val/test indices
├─ mlp_best.pt                    # best checkpoint
└─ README.md
```

## Requirements
- Python 3.10+ (tested: Python 3.13.5 on macOS, MPS; CUDA auto-detected if available).
- PyTorch (install matching your platform; see [pytorch.org](https://pytorch.org/get-started/locally/)).
- FreeGS (installed in editable mode from the bundled `freegs/` directory).
- NumPy, SciPy (optional but recommended for smoother interpolation), Matplotlib.

## Quickstart
```bash
git clone <your-repo-url>
cd freegs/FreeGS_DL

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Install PyTorch (pick command for your platform, e.g. CPU-only):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install FreeGS from the bundled source (editable):
pip install -e ..

# Project-level deps
pip install numpy scipy matplotlib
```

> `dataset_freegs/` and `image/` are already included, so you can train/eval right away. Regenerate if you want fresh data.

## Data Generation
```bash
python generate_dataset_freegs.py
# outputs to dataset_freegs/:
#   X.npy        (N, n_features) = (N, 41)
#   Y_psi.npy    (N, 65, 65)
#   meta.json    (per-sample metadata and solve status)
#   sensors.json (sensor locations)
```
Defaults: 200 samples, grid 65×65. Adjust by editing `main()` params or calling `main(out_dir=..., n_samples=..., ...)` inside the script.

## Train/Val/Test Split
```bash
python make_splits.py   # seed=42; saves splits.json
```

## Train Baseline MLP
```bash
python train_baseline_mlp.py
# uses DEVICE=mps|cuda|cpu automatically
# saves best checkpoint to mlp_best.pt (with x_mean/x_std)
```
Training normalizes inputs using train statistics and normalizes each psi target per-sample to [0,1]. Early stopping patience=30 epochs, max 400 epochs.

## Evaluate & Plot
```bash
python eval_and_plot.py
# prints mean test MSE (on normalized psi)
# shows GT vs. predicted contours for a few test samples
```
If you want files instead of pop-up windows, replace the `plt.show()` calls with `plt.savefig("image/<name>.png")` (existing figures in `image/` were produced this way).

## Notes & Known Gaps
- Baseline model is intentionally simple (fully-connected); results are still noisy. Consider CNN/U-Net or richer conditioning for better spatial fidelity.
- Randomness is only partially controlled (split seed=42; training loop has no explicit torch seed), so exact metrics may vary slightly.
- SciPy is optional; without it, interpolation falls back to nearest-neighbor and can be less smooth.

## Repro Checklist
- [ ] Confirm Python & torch versions (and whether MPS/CUDA was used).
- [ ] Keep `dataset_freegs/` under the repo or provide download instructions if you don’t want to regenerate.
- [ ] Include representative figures in `image/` (GT vs. pred) for quick inspection.
- [ ] Add a license file if publishing publicly.

