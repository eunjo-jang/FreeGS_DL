import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "dataset_freegs"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load data
X = np.load(f"{DATA_DIR}/X.npy").astype(np.float32)       # (N, 41)
Y = np.load(f"{DATA_DIR}/Y_psi.npy").astype(np.float32)   # (N, 65, 65)

with open("splits.json", "r") as f:
    splits = json.load(f)

train_idx = np.array(splits["train_idx"])
val_idx = np.array(splits["val_idx"])

# ---- Normalization (X: standardize using train only)
x_mean = X[train_idx].mean(axis=0, keepdims=True)
x_std = X[train_idx].std(axis=0, keepdims=True) + 1e-8

# Y: per-sample min-max to [0,1] (stable for training)
def normalize_psi(psi):
    mn = psi.min()
    mx = psi.max()
    return (psi - mn) / (mx - mn + 1e-8), mn, mx



class FreeGSDataset(Dataset):
    def __init__(self, idx):
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        x = (X[k:k+1] - x_mean) / x_std  # (1,41)
        psi = Y[k]                      # (65,65)
        psi_n, mn, mx = normalize_psi(psi)
        return (
            torch.from_numpy(x.squeeze(0)),               # (41,)
            torch.from_numpy(psi_n.reshape(-1)),          # (4225,)
            torch.tensor([mn, mx], dtype=torch.float32),  # (2,) for potential debug
        )

train_ds = FreeGSDataset(train_idx)
val_ds = FreeGSDataset(val_idx)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# ---- Model
class MLP(nn.Module):
    def __init__(self, in_dim=41, out_dim=65*65):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=X.shape[1]).to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
loss_fn = nn.MSELoss()

best_val = float("inf")
patience = 30
pat = 0

for epoch in range(1, 401):
    # Train
    model.train()
    train_loss = 0.0
    for xb, yb, _ in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_ds)

    # Val
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_ds)

    print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        pat = 0
        torch.save(
            {
                "model_state": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
            },
            "mlp_best.pt",
        )
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping.")
            break

print("Saved: mlp_best.pt (best_val =", best_val, ")")