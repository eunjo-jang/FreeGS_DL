import json
import numpy as np

DATA_DIR = "dataset_freegs"
SEED = 42

X = np.load(f"{DATA_DIR}/X.npy")
N = X.shape[0]

rng = np.random.default_rng(SEED)
idx = rng.permutation(N)

n_train = int(0.75 * N)
n_val = int(0.12 * N)
n_test = N - n_train - n_val

splits = {
    "seed": SEED,
    "train_idx": idx[:n_train].tolist(),
    "val_idx": idx[n_train:n_train+n_val].tolist(),
    "test_idx": idx[n_train+n_val:].tolist(),
}

with open("splits.json", "w") as f:
    json.dump(splits, f, indent=2)

print("Saved splits.json")
print("N:", N, "train:", len(splits["train_idx"]), "val:", len(splits["val_idx"]), "test:", len(splits["test_idx"]))