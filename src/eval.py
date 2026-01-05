import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset import load_data_and_splits, normalize_psi
from src.models import MLP, CoordMLP, DeepONet, PINNModel
from src.utils import get_device, load_config, parse_common_args, ensure_dir


def evaluate(cfg, ckpt_path: Path, save_dir: Path, num_examples: int = 4):
    device = get_device()
    data_dir = Path(cfg["data_dir"])
    splits_path = Path(cfg["splits_path"])

    # Load dataset + splits
    X, Y, splits = load_data_and_splits(data_dir, splits_path)
    ny, nx = Y.shape[1], Y.shape[2]
    test_idx = np.array(splits["test_idx"])

    # Restore checkpoint and normalization stats
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    x_mean = ckpt["x_mean"].astype(np.float32)
    x_std = ckpt["x_std"].astype(np.float32)

    # =========================================================
    # model name & task type
    # =========================================================
    name = cfg["model"].get("name", "mlp")
    hidden = cfg["model"].get("hidden", [256, 512, 1024])

    POINTWISE_MODELS = {"coord_mlp", "deeponet", "pinn"}
    is_pointwise = name in POINTWISE_MODELS

    # =========================================================
    # model selection
    # =========================================================
    if name == "mlp":
        model = MLP(
            in_dim=cfg["model"]["in_dim"],
            out_dim=cfg["model"]["out_dim"],
            hidden=hidden,
        ).to(device)

    elif name == "coord_mlp":
        model = CoordMLP(
            in_dim=cfg["model"]["in_dim"],
            out_dim=1,
            hidden=hidden,
        ).to(device)

    elif name == "deeponet":
        model = DeepONet(
            x_dim=cfg["model"]["in_dim"],
            latent_dim=cfg["model"]["latent_dim"],
            hidden=hidden,
        ).to(device)

    elif name == "pinn":
        model = PINNModel(
            in_dim=cfg["model"]["in_dim"],
            hidden=hidden,
        ).to(device)

    else:
        raise ValueError(f"Unknown model name: {name}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # =========================================================
    # grid definition (for pointwise models)
    # =========================================================
    if is_pointwise:
        Rmin, Rmax = cfg["generate"]["Rmin"], cfg["generate"]["Rmax"]
        Zmin, Zmax = cfg["generate"]["Zmin"], cfg["generate"]["Zmax"]
        R_grid = np.linspace(Rmin, Rmax, nx, dtype=np.float32)
        Z_grid = np.linspace(Zmin, Zmax, ny, dtype=np.float32)

    # =========================================================
    # evaluation
    # =========================================================
    mse_list = []
    relerr_list = []
    spatial_mse_acc = np.zeros((ny, nx), dtype=np.float64)
    eps = 1e-12

    with torch.no_grad():
        for k in test_idx:
            x = (X[k:k+1] - x_mean) / x_std
            psi_gt = Y[k]
            psi_gt_n, _, _ = normalize_psi(psi_gt)

            if not is_pointwise:
                pred_n = (
                    model(torch.from_numpy(x).to(device))
                    .cpu()
                    .numpy()
                    .reshape(ny, nx)
                )
            else:
                pred_n = np.zeros((ny, nx), dtype=np.float32)
                xb = torch.from_numpy(x).to(device)

                for j, Z in enumerate(Z_grid):
                    for i, R in enumerate(R_grid):
                        rz = torch.tensor([[R, Z]], device=device)
                        pred_n[j, i] = model(xb, rz).item()

            diff = pred_n - psi_gt_n
            mse = float(np.mean(diff ** 2))
            relerr = float(np.linalg.norm(diff) / (np.linalg.norm(psi_gt_n) + eps))

            spatial_mse_acc += diff ** 2
            mse_list.append(mse)
            relerr_list.append(relerr)

    mean_mse = float(np.mean(mse_list))
    spatial_mse = spatial_mse_acc / len(test_idx)

    print(
        f"[Primary] Test MSE (psi normalized): "
        f"mean={mean_mse:.6f} "
        f"min={min(mse_list):.6f} "
        f"median={np.median(mse_list):.6f} "
        f"max={max(mse_list):.6f}"
    )
    print(
        f"[RelErr ] L2 relative error: "
        f"mean={np.mean(relerr_list):.6f} "
        f"min={min(relerr_list):.6f} "
        f"median={np.median(relerr_list):.6f} "
        f"max={max(relerr_list):.6f}"
    )
    print(
        f"[Spatial] Spatial MSE map stats: "
        f"mean={spatial_mse.mean():.6f} "
        f"min={spatial_mse.min():.6f} "
        f"median={np.median(spatial_mse):.6f} "
        f"max={spatial_mse.max():.6f}"
    )

    # =========================================================
    # plotting
    # =========================================================
    ensure_dir(save_dir)
    show_n = min(num_examples, len(test_idx))

    for i in range(show_n):
        k = int(test_idx[i])
        x = (X[k:k+1] - x_mean) / x_std
        psi_gt = Y[k]
        psi_gt_n, _, _ = normalize_psi(psi_gt)

        with torch.no_grad():
            if not is_pointwise:
                pred_n = (
                    model(torch.from_numpy(x).to(device))
                    .cpu()
                    .numpy()
                    .reshape(ny, nx)
                )
            else:
                pred_n = np.zeros((ny, nx), dtype=np.float32)
                xb = torch.from_numpy(x).to(device)
                for j, Z in enumerate(Z_grid):
                    for i_, R in enumerate(R_grid):
                        rz = torch.tensor([[R, Z]], device=device)
                        pred_n[j, i_] = model(xb, rz).item()

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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--num-examples", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint or cfg["checkpoint_path"])
    save_dir = Path(args.save_dir or cfg["image_out_dir"])

    evaluate(cfg, ckpt_path, save_dir, num_examples=args.num_examples)


if __name__ == "__main__":
    main()