# generate_dataset_freegs.py
import os
import json
import numpy as np

import freegs

# (선택) scipy 있으면 2D 보간이 깔끔함
try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def make_sensor_layout():
    """
    '기본 센서' 세팅(임의 고정):
    - flux loops 8개
    - magnetic probes 16개 (BR,BZ)
    - rogowski 1개 (Ip)
    """
    # 대충 벽 근처에 분포시키는 예시 (TestTokamak 도메인 기준)
    flux_loops = [
        (1.75,  0.20), (1.75, -0.20),
        (1.55,  0.70), (1.55, -0.70),
        (1.05,  0.80), (1.05, -0.80),
        (0.85,  0.30), (0.85, -0.30),
    ]

    probes = []
    for z in np.linspace(-0.85, 0.85, 8):
        probes.append((0.75, z))   # inboard-ish
        probes.append((1.80, z))   # outboard-ish
    # probes: 16개 (각각 BR,BZ를 뽑음)

    return {"flux_loops": flux_loops, "probes": probes}


def interp2d(r1d, z1d, field2d):
    """(R,Z)->field 보간 함수 생성"""
    if _HAS_SCIPY:
        itp = RegularGridInterpolator((z1d, r1d), field2d, bounds_error=False, fill_value=np.nan)
        return lambda R, Z: float(itp([[Z, R]])[0])
    # scipy 없으면 nearest
    def nn(R, Z):
        ir = int(np.argmin(np.abs(r1d - R)))
        iz = int(np.argmin(np.abs(z1d - Z)))
        return float(field2d[iz, ir])
    return nn


def compute_BR_BZ(r1d, z1d, psi2d):
    """
    BR = -1/R * dpsi/dZ
    BZ =  1/R * dpsi/dR
    """
    dpsi_dz, dpsi_dr = np.gradient(psi2d, z1d, r1d)  # shape: (nz, nr)
    Rmesh = np.tile(r1d.reshape(1, -1), (len(z1d), 1))
    BR = -(1.0 / (Rmesh + 1e-12)) * dpsi_dz
    BZ =  (1.0 / (Rmesh + 1e-12)) * dpsi_dr
    return BR, BZ


def sample_case(rng):
    """랜덤 케이스(프로파일/목표) 샘플링: 너무 넓게 잡지 말고 안정 범위부터 시작"""
    paxis = float(rng.uniform(5e2, 5e3))    # Pa
    Ip    = float(rng.uniform(1e5, 6e5))    # A
    fvac  = float(rng.uniform(0.8, 2.5))    # vacuum f = R*Bt (스케일 파라미터)

    # X-point 목표도 약간 흔들기 (01-freeboundary.py 근처)
    x1 = (float(rng.uniform(1.05, 1.20)), float(rng.uniform(-0.75, -0.45)))
    x2 = (float(rng.uniform(1.05, 1.20)), float(rng.uniform( 0.65,  0.95)))

    # isoflux도 하나만 (간단 버전)
    isoflux = [(x1[0], x1[1], x1[0], -x1[1])]

    return {"paxis": paxis, "Ip": Ip, "fvac": fvac, "xpoints": [x1, x2], "isoflux": isoflux}


def main(
    out_dir="dataset_freegs",
    n_samples=200,
    seed=0,
    nx=65, ny=65,
    Rmin=0.1, Rmax=2.0,
    Zmin=-1.0, Zmax=1.0,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    sensors = make_sensor_layout()
    with open(os.path.join(out_dir, "sensors.json"), "w") as f:
        json.dump(sensors, f, indent=2)

    X_list = []
    Ypsi_list = []
    meta = []

    for i in range(n_samples):
        case = sample_case(rng)

        # 1) 머신/격자 생성
        tokamak = freegs.machine.TestTokamak()
        eq = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=Rmin, Rmax=Rmax,
            Zmin=Zmin, Zmax=Zmax,
            nx=nx, ny=ny,
            boundary=freegs.boundary.freeBoundaryHagenow,
        )

        # 2) 프로파일(내부 레시피) 지정: "모양은 고정, 총량(paxis, Ip) 맞추는" 방식
        profiles = freegs.jtor.ConstrainPaxisIp(eq, case["paxis"], case["Ip"], case["fvac"])

        # 3) 형상 제약(목표) 지정 -> 코일 전류를 자동으로 찾아감
        constrain = freegs.control.constrain(
            xpoints=case["xpoints"],
            isoflux=case["isoflux"],
        )

        # 4) Solve (Picard 반복)
        try:
            freegs.solve(eq, profiles, constrain, show=False)
        except Exception as e:
            # 실패 케이스는 스킵 (처음엔 이런게 많을 수 있음)
            meta.append({**case, "ok": False, "error": str(e)})
            continue

        # 5) 정답: psi(R,Z) 저장
        # FreeGS에서 psi 접근은 보통 eq.psi() 형태 (버전에 따라 다르면 여기만 수정)
        psi2d = eq.psi()

        # 6) 센서값(입력 X) 만들기: psi 보간 + 미분해서 BR/BZ
        # R,Z 1D 축도 버전에 따라 이름이 다를 수 있어서, 가능한 속성 먼저 시도
        if hasattr(eq, "R_1D") and hasattr(eq, "Z_1D"):
            r1d = np.asarray(eq.R_1D)
            z1d = np.asarray(eq.Z_1D)
        else:
            # fallback: eq.R, eq.Z가 mesh면 unique로 뽑기
            r1d = np.unique(np.asarray(eq.R).ravel())
            z1d = np.unique(np.asarray(eq.Z).ravel())

        BR2d, BZ2d = compute_BR_BZ(r1d, z1d, psi2d)
        psi_itp = interp2d(r1d, z1d, psi2d)
        br_itp  = interp2d(r1d, z1d, BR2d)
        bz_itp  = interp2d(r1d, z1d, BZ2d)

        x_feats = []

        # flux loops: psi
        for (R, Z) in sensors["flux_loops"]:
            x_feats.append(psi_itp(R, Z))

        # probes: BR, BZ
        for (R, Z) in sensors["probes"]:
            x_feats.append(br_itp(R, Z))
            x_feats.append(bz_itp(R, Z))

        # rogowski: Ip (여기서는 케이스의 Ip를 "측정값"으로 간주)
        x_feats.append(case["Ip"])

        x_feats = np.asarray(x_feats, dtype=np.float32)

        X_list.append(x_feats)
        Ypsi_list.append(np.asarray(psi2d, dtype=np.float32))
        meta.append({**case, "ok": True})

    X = np.stack(X_list, axis=0)
    Ypsi = np.stack(Ypsi_list, axis=0)

    np.save(os.path.join(out_dir, "X.npy"), X)          # (N, n_features)
    np.save(os.path.join(out_dir, "Y_psi.npy"), Ypsi)   # (N, ny, nx)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_dir}")
    print(f"X: {X.shape}  Y_psi: {Ypsi.shape}")
    print(f"n_features = {X.shape[1]}")


if __name__ == "__main__":
    main()