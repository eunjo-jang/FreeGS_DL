import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset import load_data_and_splits, normalize_psi
from src.model import MLP
from src.utils import get_device, load_config, parse_common_args, ensure_dir


def evaluate(cfg, ckpt_path: Path, save_dir: Path, num_examples: int = 4):
    device = get_device()
    data_dir = Path(cfg["data_dir"])
    splits_path = Path(cfg["splits_path"])

    X, Y, splits = load_data_and_splits(data_dir, splits_path)
    test_idx = np.array(splits["test_idx"])

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    x_mean = ckpt["x_mean"].astype(np.float32)
    x_std = ckpt["x_std"].astype(np.float32)

    model = MLP(in_dim=cfg["model"]["in_dim"], out_dim=cfg["model"]["out_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mse_list = []
    with torch.no_grad():
        for k in test_idx:
            x = (X[k:k+1] - x_mean) / x_std
            psi_gt = Y[k]
            psi_gt_n, mn, mx = normalize_psi(psi_gt)
            pred_n = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(65, 65)
            mse = float(np.mean((pred_n - psi_gt_n) ** 2))
            mse_list.append(mse)

    mean_mse = float(np.mean(mse_list))
    print(f"Test MSE (psi normalized): mean={mean_mse:.6f} min={min(mse_list):.6f} median={np.median(mse_list):.6f} max={max(mse_list):.6f}")

    ensure_dir(save_dir)
    show_n = min(num_examples, len(test_idx))
    for i in range(show_n):
        k = int(test_idx[i])
        x = (X[k:k+1] - x_mean) / x_std
        psi_gt = Y[k]
        psi_gt_n, _, _ = normalize_psi(psi_gt)
        with torch.no_grad():
            pred_n = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(65, 65)

        fig = plt.figure()
        plt.contour(psi_gt_n, levels=30)
        plt.title(f"GT psi (norm), sample {k}")
        plt.gca().set_aspect("equal")
        fig.savefig(save_dir / f"gt_{k}.png", bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.contour(pred_n, levels=30)
        plt.title(f"Pred psi (norm), sample {k}")
        plt.gca().set_aspect("equal")
        fig.savefig(save_dir / f"pred_{k}.png", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = parse_common_args("Evaluate and plot")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--save-dir", default=None, help="Where to save images")
    parser.add_argument("--num-examples", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint or cfg["checkpoint_path"])
    save_dir = Path(args.save_dir or cfg["image_out_dir"])

    evaluate(cfg, ckpt_path, save_dir, num_examples=args.num_examples)


if __name__ == "__main__":
    main()

