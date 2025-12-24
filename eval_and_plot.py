import json
import numpy as np
import torch
import matplotlib.pyplot as plt

DATA_DIR = "dataset_freegs"
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

X = np.load(f"{DATA_DIR}/X.npy").astype(np.float32)
Y = np.load(f"{DATA_DIR}/Y_psi.npy").astype(np.float32)

with open("splits.json", "r") as f:
    splits = json.load(f)
test_idx = np.array(splits["test_idx"])

ckpt = torch.load("mlp_best.pt", map_location=DEVICE, weights_only=False)
x_mean = ckpt["x_mean"].astype(np.float32)
x_std = ckpt["x_std"].astype(np.float32)

# Model must match training
import torch.nn as nn

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
model.load_state_dict(ckpt["model_state"])
model.eval()

def normalize_psi(psi):
    mn = psi.min()
    mx = psi.max()
    return (psi - mn) / (mx - mn + 1e-8), mn, mx

mse_list = []

# Evaluate
with torch.no_grad():
    for k in test_idx:
        x = (X[k:k+1] - x_mean) / x_std
        psi_gt = Y[k]

        psi_gt_n, mn, mx = normalize_psi(psi_gt)

        pred_n = model(torch.from_numpy(x).to(DEVICE)).cpu().numpy().reshape(65, 65)

        mse = np.mean((pred_n - psi_gt_n) ** 2)
        mse_list.append(mse)

print("Test MSE (psi normalized):", float(np.mean(mse_list)))

# Plot a few examples
show_n = min(4, len(test_idx))
for i in range(show_n):
    k = int(test_idx[i])
    x = (X[k:k+1] - x_mean) / x_std
    psi_gt = Y[k]
    psi_gt_n, mn, mx = normalize_psi(psi_gt)

    with torch.no_grad():
        pred_n = model(torch.from_numpy(x).to(DEVICE)).cpu().numpy().reshape(65, 65)

    plt.figure()
    plt.contour(psi_gt_n, levels=30)
    plt.title(f"GT psi (norm), sample {k}")
    plt.gca().set_aspect("equal")
    plt.show()

    plt.figure()
    plt.contour(pred_n, levels=30)
    plt.title(f"Pred psi (norm), sample {k}")
    plt.gca().set_aspect("equal")
    plt.show()