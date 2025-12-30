import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_psi(psi: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mn = float(psi.min())
    mx = float(psi.max())
    return (psi - mn) / (mx - mn + 1e-8), mn, mx


class FreeGSDataset(Dataset):
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
    X = np.load(data_dir / "X.npy").astype(np.float32)
    Y = np.load(data_dir / "Y_psi.npy").astype(np.float32)
    with open(splits_path, "r") as f:
        splits = json.load(f)
    return X, Y, splits

