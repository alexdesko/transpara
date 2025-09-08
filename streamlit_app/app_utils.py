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
from pytorch_grad_cam import GradCAM

# Ensure the project src/ is importable for `dataio`, `models`, `utils`
APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
SRC_DIR = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


from dataio import custom_transform
from models import CustomResNet18




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
    """Construct a model (ResNet18 only)."""
    if model_name != "ResNet18":
        raise ValueError("Only ResNet18 is supported in this app.")
    return CustomResNet18(num_classes=num_classes)


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
    model.eval()
    return model, cfg


def preprocess_image(img: Image.Image, input_size: int) -> torch.Tensor:
    """To tensor, resize, normalize; returns (1,C,H,W) on device."""
    # Use RGB to be compatible with ResNet18 default transforms (3-channel)
    x = custom_transform()(img.convert("RGB")).unsqueeze(0)
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
    print('went there already')
    # Access base for torchvision resnet
    base = model.model if hasattr(model, "model") else model
    # Try a reasonable default target conv for ResNet18
    try:
        target = base.layer4[-1].conv2
    except Exception as e:
        raise RuntimeError("Grad-CAM currently supports ResNet18 models only in this demo.") from e

    model.unfreeze_layers()
    print('before pre-process')
    input_tensor = preprocess_image(img, input_size)
    print('after pre-process')

    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=target) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs

    return cam.cpu().numpy(), cls
