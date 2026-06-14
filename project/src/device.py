"""
Safe Torch device selection.

Some laptop GPUs (e.g. Blackwell-generation RTX 50-series, sm_120) report
``torch.cuda.is_available() == True`` even when the installed PyTorch build
does not ship kernels for that compute capability. Blindly trusting
``is_available()`` then crashes on the first CUDA kernel launch. This module
runs a tiny real kernel launch and silently falls back to CPU if it fails,
so every training script in the project can call ``get_device()`` safely.
"""

import torch

_DEVICE = None


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            x = torch.randn(2, 2, device="cuda")
            _ = (x @ x).sum().item()
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")

    _DEVICE = device
    print(f"[device] Using {device}")
    return device
