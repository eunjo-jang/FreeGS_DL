import json
from pathlib import Path
import numpy as np
from src.utils import load_config, parse_common_args, ensure_dir


def main():
    parser = parse_common_args("Create train/val/test splits")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir = cfg["data_dir"]
    splits_path = cfg["splits_path"]

    X = np.load(f"{data_dir}/X.npy")
    N = X.shape[0]

    rng = np.random.default_rng(cfg["train"]["seed"])
    idx = rng.permutation(N)

    n_train = int(0.75 * N)
    n_val = int(0.12 * N)
    n_test = N - n_train - n_val

    splits = {
        "seed": cfg["train"]["seed"],
        "train_idx": idx[:n_train].tolist(),
        "val_idx": idx[n_train:n_train + n_val].tolist(),
        "test_idx": idx[n_train + n_val:].tolist(),
    }

    # ensure target directory exists
    ensure_dir(str(Path(splits_path).parent))
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved splits to {splits_path}")
    print(f"N={N} train={len(splits['train_idx'])} val={len(splits['val_idx'])} test={len(splits['test_idx'])}")


if __name__ == "__main__":
    main()

