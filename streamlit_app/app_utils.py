"""Shared utilities for the Streamlit demo app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from dataio.transform import custom_transform
from models import CustomResNet18, SimpleCNN, SimpleMLP
from utils.device import get_device

# Ensure the project src/ is importable for `dataio`, `models`, `utils`
APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
SRC_DIR = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


RUN_BASES = [Path("outputs"), Path("trained_models")]


def find_runs(bases: Optional[Iterable[Path]] = None) -> list[Path]:
    """Return sorted list of run directories containing a metrics.csv.

    Searches typical training output roots (Hydra `outputs/` and legacy
    `trained_models/`). Returns runs sorted by modification time (newest first).
    """
    bases = list(bases) if bases is not None else RUN_BASES
    candidates: list[Tuple[Path, float]] = []
    for base in bases:
        if not base.exists():
            continue
        for d in base.rglob("*"):
            if d.is_dir() and (d / "metrics.csv").exists():
                try:
                    mtime = (d / "metrics.csv").stat().st_mtime
                except OSError:
                    continue
                candidates.append((d, mtime))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates]


def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load metrics.csv from a run directory."""
    return pd.read_csv(run_dir / "metrics.csv")


def load_config(run_dir: Path) -> DictConfig:
    """Load YAML config from a run directory (if present)."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return OmegaConf.create()  # empty DictConfig
    return OmegaConf.load(cfg_path)  # loads directly as DictConfig


def build_model(model_name: str, input_size: int, num_classes: int):
    """Construct a model by name consistent with training code."""
    if model_name == "ResNet18":
        model = CustomResNet18(num_classes=num_classes, in_channels=1, weights=None)
    elif model_name == "CNN":
        model = SimpleCNN(input_size=input_size, num_classes=num_classes)
    elif model_name == "MLP":
        model = SimpleMLP(input_size=input_size * input_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def load_model_from_run(run_dir: Path):
    """Load model.pth + config.yaml and return (model, cfg)."""
    cfg = load_config(run_dir)
    # Backwards-compat for flat config
    print(OmegaConf.select(cfg, "model.name"))
    model_name, input_size, num_classes = [
        OmegaConf.select(cfg, key) for key in ("model.name", "data.input_size", "model.num_classes")
    ]

    model = build_model(model_name, input_size, num_classes)
    state = torch.load(run_dir / "model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(get_device())
    return model, cfg


def preprocess_image(img: Image.Image, input_size: int) -> torch.Tensor:
    """To tensor, resize, normalize; returns (1,C,H,W) on device."""
    x = custom_transform(input_size)(img.convert("L")).unsqueeze(0).to(get_device())
    return x


def predict_probs(model: torch.nn.Module, img: Image.Image, input_size: int) -> np.ndarray:
    """Run model and return softmax probabilities as numpy array of shape (num_classes,)."""
    x = preprocess_image(img, input_size)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def gradcam(model: torch.nn.Module, img: Image.Image, input_size: int) -> tuple[np.ndarray, int]:
    """Compute a simple Grad-CAM heatmap for ResNet-like models.

    Returns (cam, pred_class) where cam is in [0,1] with shape (H,W) in the
    last conv's spatial resolution, which should be resized by the caller.
    """
    # Access base for torchvision resnet
    base = model.model if hasattr(model, "model") else model
    # Try a reasonable default target conv for ResNet18
    try:
        target = base.layer4[-1].conv2
    except Exception as e:
        raise RuntimeError("Grad-CAM currently supports ResNet18 models only in this demo.") from e

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_, __, out):
        activations.append(out.detach())

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    x = preprocess_image(img, input_size)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    cls = int(logits.argmax(1).item())
    logits[:, cls].backward()

    acts = activations[-1][0]  # (C,H,W)
    grads = gradients[-1][0]  # (C,H,W)
    weights = grads.mean(dim=(1, 2))  # (C,)
    cam = (weights[:, None, None] * acts).sum(0)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-6)

    h1.remove()
    h2.remove()
    return cam.cpu().numpy(), cls
