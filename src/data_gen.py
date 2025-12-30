import os
import json
import numpy as np

import freegs
from src.utils import ensure_dir, load_config, parse_common_args

# optional interpolation
try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def make_sensor_layout():
    flux_loops = [
        (1.75, 0.20), (1.75, -0.20),
        (1.55, 0.70), (1.55, -0.70),
        (1.05, 0.80), (1.05, -0.80),
        (0.85, 0.30), (0.85, -0.30),
    ]
    probes = []
    for z in np.linspace(-0.85, 0.85, 8):
        probes.append((0.75, z))
        probes.append((1.80, z))
    return {"flux_loops": flux_loops, "probes": probes}


def interp2d(r1d, z1d, field2d):
    if _HAS_SCIPY:
        itp = RegularGridInterpolator((z1d, r1d), field2d, bounds_error=False, fill_value=np.nan)
        return lambda R, Z: float(itp([[Z, R]])[0])

    def nn(R, Z):
        ir = int(np.argmin(np.abs(r1d - R)))
        iz = int(np.argmin(np.abs(z1d - Z)))
        return float(field2d[iz, ir])
    return nn


def compute_BR_BZ(r1d, z1d, psi2d):
    dpsi_dz, dpsi_dr = np.gradient(psi2d, z1d, r1d)
    Rmesh = np.tile(r1d.reshape(1, -1), (len(z1d), 1))
    BR = -(1.0 / (Rmesh + 1e-12)) * dpsi_dz
    BZ = (1.0 / (Rmesh + 1e-12)) * dpsi_dr
    return BR, BZ


def sample_case(rng):
    paxis = float(rng.uniform(5e2, 5e3))
    Ip = float(rng.uniform(1e5, 6e5))
    fvac = float(rng.uniform(0.8, 2.5))
    x1 = (float(rng.uniform(1.05, 1.20)), float(rng.uniform(-0.75, -0.45)))
    x2 = (float(rng.uniform(1.05, 1.20)), float(rng.uniform(0.65, 0.95)))
    isoflux = [(x1[0], x1[1], x1[0], -x1[1])]
    return {"paxis": paxis, "Ip": Ip, "fvac": fvac, "xpoints": [x1, x2], "isoflux": isoflux}


def main():
    parser = parse_common_args("Generate synthetic FreeGS dataset")
    args = parser.parse_args()
    cfg = load_config(args.config)
    gcfg = cfg["generate"]
    out_dir = cfg["data_dir"]

    ensure_dir(out_dir)
    rng = np.random.default_rng(gcfg["seed"])

    sensors = make_sensor_layout()
    with open(os.path.join(out_dir, "sensors.json"), "w") as f:
        json.dump(sensors, f, indent=2)

    X_list, Ypsi_list, meta = [], [], []

    for _ in range(gcfg["n_samples"]):
        case = sample_case(rng)
        tokamak = freegs.machine.TestTokamak()
        eq = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=gcfg["Rmin"], Rmax=gcfg["Rmax"],
            Zmin=gcfg["Zmin"], Zmax=gcfg["Zmax"],
            nx=gcfg["nx"], ny=gcfg["ny"],
            boundary=freegs.boundary.freeBoundaryHagenow,
        )
        profiles = freegs.jtor.ConstrainPaxisIp(eq, case["paxis"], case["Ip"], case["fvac"])
        constrain = freegs.control.constrain(xpoints=case["xpoints"], isoflux=case["isoflux"])

        try:
            freegs.solve(eq, profiles, constrain, show=False)
        except Exception as e:
            meta.append({**case, "ok": False, "error": str(e)})
            continue

        psi2d = eq.psi()
        if hasattr(eq, "R_1D") and hasattr(eq, "Z_1D"):
            r1d = np.asarray(eq.R_1D)
            z1d = np.asarray(eq.Z_1D)
        else:
            r1d = np.unique(np.asarray(eq.R).ravel())
            z1d = np.unique(np.asarray(eq.Z).ravel())

        BR2d, BZ2d = compute_BR_BZ(r1d, z1d, psi2d)
        psi_itp = interp2d(r1d, z1d, psi2d)
        br_itp = interp2d(r1d, z1d, BR2d)
        bz_itp = interp2d(r1d, z1d, BZ2d)

        x_feats = []
        for (R, Z) in sensors["flux_loops"]:
            x_feats.append(psi_itp(R, Z))
        for (R, Z) in sensors["probes"]:
            x_feats.append(br_itp(R, Z))
            x_feats.append(bz_itp(R, Z))
        x_feats.append(case["Ip"])

        X_list.append(np.asarray(x_feats, dtype=np.float32))
        Ypsi_list.append(np.asarray(psi2d, dtype=np.float32))
        meta.append({**case, "ok": True})

    X = np.stack(X_list, axis=0)
    Ypsi = np.stack(Ypsi_list, axis=0)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "Y_psi.npy"), Ypsi)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {out_dir}")
    print(f"X: {X.shape}  Y_psi: {Ypsi.shape}")
    print(f"n_features = {X.shape[1]}")


if __name__ == "__main__":
    main()

