"""
Simple disk cache for preprocessed signals and spectral results.
Saves numpy arrays to .npz files so phases 1-3 only run once.
"""

import numpy as np
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "outputs" / ".cache"


def _prep_path():
    return CACHE_DIR / "preprocessed.pkl"


def _spec_path():
    return CACHE_DIR / "spectral.pkl"


def save_preprocessed(data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_prep_path(), "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] Preprocessed saved → {_prep_path()}")


def load_preprocessed() -> dict | None:
    p = _prep_path()
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    print(f"[cache] Loaded preprocessed from cache ({len(data)} files)")
    return data


def save_spectral(data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # spectral contains numpy arrays — pickle handles them fine
    with open(_spec_path(), "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] Spectral saved → {_spec_path()}")


def load_spectral() -> dict | None:
    p = _spec_path()
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    print(f"[cache] Loaded spectral from cache ({len(data)} files)")
    return data
