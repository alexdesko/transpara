from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from app_utils import find_runs, gradcam, load_model_from_run
from matplotlib import cm
from PIL import Image
from skimage.transform import resize as imresize

# Ensure local utils take precedence over any installed package named "streamlit_app"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Explainability", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_model(run_dir: Path):
    model, cfg = load_model_from_run(run_dir)
    return model, cfg


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Overlay a heatmap CAM onto the input image."""
    base = np.array(img.convert("RGB")) / 255.0
    cam_resized = imresize(cam, (base.shape[0], base.shape[1]), preserve_range=True)
    heat = cm.jet(cam_resized)[..., :3]  # RGBA -> RGB
    out = (1 - alpha) * base + alpha * heat
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def main():
    st.header("Explainability (Grad‑CAM)")

    runs = find_runs()
    if not runs:
        st.warning("No runs found under outputs/ or trained_models/.")
        return

    run = st.selectbox("Select a run (ResNet18 recommended)", options=runs, format_func=lambda p: str(p))
    model, cfg = _load_model(run)
    model_name = cfg.get("model", {}).get("name")

    if model_name != "ResNet18":
        st.info("Grad‑CAM demo currently supports ResNet18 models.")

    uploaded = st.file_uploader("Upload a chest X‑ray image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    img = Image.open(uploaded)
    col1, col2 = st.columns(2)
    col1.image(img, caption="Input", use_container_width=True)

    input_size = int(cfg.get("data", {}).get("input_size") or cfg.get("input_size", 256))
    try:
        cam, cls = gradcam(model, img, input_size)
        alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.35, 0.05)
        overlay = overlay_cam(img, cam, alpha=alpha)
        col2.image(overlay, caption=f"Grad‑CAM (class={cls})", use_container_width=True)
    except RuntimeError as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
