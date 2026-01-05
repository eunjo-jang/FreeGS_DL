import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_psi(psi: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Min-max normalize psi; return normalized field and original min/max."""
    mn = float(psi.min())
    mx = float(psi.max())
    return (psi - mn) / (mx - mn + 1e-8), mn, mx

class FreeGSDataset(Dataset):
    """Grid-based dataset: sensor vector -> flattened psi grid."""
    def __init__(self, X: np.ndarray, Y: np.ndarray, idx: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray):
        self.X = X
        self.Y = Y
        self.idx = idx
        self.x_mean = x_mean
        self.x_std = x_std

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        x = (self.X[k:k+1] - self.x_mean) / self.x_std
        psi = self.Y[k]
        psi_n, mn, mx = normalize_psi(psi)
        return (
            torch.from_numpy(x.squeeze(0)),
            torch.from_numpy(psi_n.reshape(-1)),
            torch.tensor([mn, mx], dtype=torch.float32),
        )


def load_data_and_splits(data_dir: Path, splits_path: Path):
    """Load dataset arrays and split indices."""
    X = np.load(data_dir / "X.npy").astype(np.float32)
    Y = np.load(data_dir / "Y_psi.npy").astype(np.float32)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    return X, Y, splits


class FreeGSDatasetCoord(Dataset):
    """Pointwise dataset: (sensor vector, RZ coord) -> psi(R,Z) scalar."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        idx: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        R_grid: np.ndarray,
        Z_grid: np.ndarray,
        points_per_sample: int = 1024,
    ):
        self.X = X
        self.Y = Y
        self.idx = idx
        self.x_mean = x_mean
        self.x_std = x_std

        self.R = R_grid
        self.Z = Z_grid
        self.points_per_sample = points_per_sample

        self.H, self.W = Y.shape[1], Y.shape[2]

    def __len__(self):
        # number of (equilibrium, grid-point) samples
        return len(self.idx) * self.points_per_sample

    def __getitem__(self, i):
        # 1) select equilibrium index
        sample_idx = self.idx[i // self.points_per_sample]

        # 2) randomly sample a grid point
        j = np.random.randint(0, self.H)
        k = np.random.randint(0, self.W)

        # 3) normalize sensor input
        x = (self.X[sample_idx:sample_idx + 1] - self.x_mean) / self.x_std

        # 4) coordinate input (R, Z)
        rz = np.array([self.R[k], self.Z[j]], dtype=np.float32)

        # 5) normalize psi (sample-wise)
        psi = self.Y[sample_idx]
        psi_n, _, _ = normalize_psi(psi)
        y = psi_n[j, k]

        return (
            torch.from_numpy(x.squeeze(0)),   # (41,)
            torch.from_numpy(rz),              # (2,)
            torch.tensor(y, dtype=torch.float32),  # scalar
        )