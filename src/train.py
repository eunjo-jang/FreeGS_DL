import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import FreeGSDataset, load_data_and_splits
from src.model import MLP
from src.utils import get_device, load_config, parse_common_args, seed_everything, ensure_dir


def main():
    parser = parse_common_args("Train baseline MLP")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    device = get_device()
    seed_everything(train_cfg["seed"])

    data_dir = Path(cfg["data_dir"])
    splits_path = Path(cfg["splits_path"])
    ckpt_path = Path(args.checkpoint or cfg["checkpoint_path"])

    X, Y, splits = load_data_and_splits(data_dir, splits_path)
    train_idx = np.array(splits["train_idx"])
    val_idx = np.array(splits["val_idx"])

    x_mean = X[train_idx].mean(axis=0, keepdims=True)
    x_std = X[train_idx].std(axis=0, keepdims=True) + 1e-8

    train_ds = FreeGSDataset(X, Y, train_idx, x_mean, x_std)
    val_ds = FreeGSDataset(X, Y, val_idx, x_mean, x_std)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["val_batch_size"], shuffle=False)

    model = MLP(in_dim=cfg["model"]["in_dim"], out_dim=cfg["model"]["out_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience = train_cfg["patience"]
    pat = 0

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            pat = 0
            ensure_dir(str(ckpt_path.parent))
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "config": cfg,
                },
                ckpt_path,
            )
        else:
            pat += 1
            if pat >= patience:
                print("Early stopping.")
                break

    print(f"Saved best checkpoint to {ckpt_path} (best_val={best_val:.6f})")


if __name__ == "__main__":
    main()

