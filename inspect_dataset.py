import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "dataset_freegs"

X = np.load(f"{DATA_DIR}/X.npy")          # (N, 41)
Y = np.load(f"{DATA_DIR}/Y_psi.npy")      # (N, 65, 65)

with open(f"{DATA_DIR}/sensors.json", "r") as f:
    sensors = json.load(f)

n_fl = len(sensors["flux_loops"])
n_pr = len(sensors["probes"])
n_feat = X.shape[1]

print("X shape:", X.shape, "Y_psi shape:", Y.shape)
print("n_flux_loops:", n_fl, "n_probes:", n_pr, "n_features:", n_feat)

# Guess feature layout
# Typical: [FL(8)] + [BR(16)] + [BZ(16)] + [Ip(1)] = 41
rest = n_feat - n_fl
print("After flux loops, remaining features:", rest)
if rest == 2 * n_pr + 1:
    print("Likely layout: [FL] + [BR probes] + [BZ probes] + [Ip]")
elif rest == 2 * n_pr:
    print("Likely layout: [FL] + [BR probes] + [BZ probes]")
else:
    print("Layout is non-standard. You'll want to check your generator ordering.")

# Basic sanity checks
print("X NaN:", np.isnan(X).any(), "Y NaN:", np.isnan(Y).any())
print("X min/max:", float(np.min(X)), float(np.max(X)))
print("Y min/max:", float(np.min(Y)), float(np.max(Y)))

# Plot 3 samples of psi
for idx in [0, min(1, len(Y)-1), min(2, len(Y)-1)]:
    psi = Y[idx]
    plt.figure()
    plt.contour(psi, levels=30)  # index grid (not physical R,Z yet)
    plt.title(f"Sample {idx}: psi contours")
    plt.gca().set_aspect("equal")
    plt.show()