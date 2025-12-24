# check_case_ab.py
import json
from pathlib import Path
import numpy as np

def extract_success_flags(meta):
    """
    meta.json 포맷이 조금 달라도 success 플래그를 최대한 찾아서 list[bool]로 반환
    """
    flags = []

    if isinstance(meta, dict):
        # (1) success 같은 키에 리스트로 저장된 케이스
        for k in ["success", "successful", "ok", "valid", "is_valid"]:
            if k in meta and isinstance(meta[k], list):
                return [bool(x) for x in meta[k]]

        # (2) samples/records에 per-sample dict로 저장된 케이스
        for list_key in ["samples", "records"]:
            if list_key in meta and isinstance(meta[list_key], list):
                for s in meta[list_key]:
                    if isinstance(s, dict):
                        for k in ["success", "successful", "ok", "valid", "is_valid"]:
                            if k in s:
                                flags.append(bool(s[k]))
                                break
                if flags:
                    return flags

    if isinstance(meta, list):
        # (3) meta 자체가 list[dict] or list[bool] 인 케이스
        for s in meta:
            if isinstance(s, bool):
                flags.append(s)
            elif isinstance(s, dict):
                for k in ["success", "successful", "ok", "valid", "is_valid"]:
                    if k in s:
                        flags.append(bool(s[k]))
                        break
        if flags:
            return flags

    return None


def main(dataset_dir="dataset_freegs"):
    d = Path(dataset_dir)
    meta_path = d / "meta.json"
    X_path = d / "X.npy"
    Y_path = d / "Y_psi.npy"

    meta = json.loads(meta_path.read_text())
    X = np.load(X_path)
    Y = np.load(Y_path)

    flags = extract_success_flags(meta)

    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")

    if flags is None:
        print("meta.json에서 success flags를 못 찾았음 → 판별 불가(메타 포맷 확인 필요)")
        return

    n_meta = len(flags)
    n_true = sum(flags)
    n_false = n_meta - n_true
    n_xy = X.shape[0]

    print(f"meta flags count = {n_meta}  (true={n_true}, false={n_false})")
    print(f"X/Y sample count = {n_xy}")

    if n_meta > n_xy:
        print("✅ 결론: Case A (meta는 전체 시도 로그, X/Y는 성공 샘플만 저장)")
    elif n_meta == n_xy:
        if n_false > 0:
            print("✅ 결론: Case B (실패 포함 1:1 저장이거나, 실패 샘플도 X/Y에 포함)")
            print("   → 실패 샘플이 X/Y에 들어갔다면, flags로 마스킹해서 걸러 쓰면 됨.")
        else:
            print("✅ 결론: Case B (meta도 성공만 / X/Y도 성공만, 1:1)")
    else:
        print("⚠️ 이상: meta flags가 X/Y보다 적음 → 저장 로직/파일 버전 불일치 가능")

if __name__ == "__main__":
    main("dataset_freegs")