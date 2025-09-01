from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available device in order: mps, cuda, cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

