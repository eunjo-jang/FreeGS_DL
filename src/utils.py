import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Read YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Select best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility (non-deterministic ops still allowed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism can reduce performance; keep False unless needed
    torch.use_deterministic_algorithms(False)


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    """Write dict-like object as pretty JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_common_args(description: str) -> argparse.ArgumentParser:
    """Base CLI parser shared by scripts (expects --config)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/mlp.yaml", help="Path to config YAML")
    return parser



