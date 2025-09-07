from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from app_utils import find_runs, load_model_from_run, predict_probs
from omegaconf import OmegaConf
from PIL import Image

# Ensure local utils take precedence over any installed package named "streamlit_app"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Try Model", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_model(run_dir: Path):
    model, cfg = load_model_from_run(run_dir)
    return model, cfg


def main():
    st.header("Try Model")

    runs = find_runs()
    if not runs:
        st.warning("No runs found under outputs/ or trained_models/.")
        return

    run = st.selectbox("Select a run", options=runs, format_func=lambda p: str(p))
    model, cfg = _load_model(run)

    st.caption("Model config")
    st.json(OmegaConf.to_container(cfg, resolve=True))

    uploaded = st.file_uploader("Upload a chest X‑ray image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    img = Image.open(uploaded)
    st.image(img, caption="Input", use_container_width=False, width=384)

    input_size = int(cfg.get("data", {}).get("input_size") or cfg.get("input_size", 256))
    probs = predict_probs(model, img, input_size)

    classes = list(cfg.get("class_names") or ["COVID", "NORMAL", "PNEUMONIA"])
    st.subheader("Predictions")
    for c, p in zip(classes, probs):
        st.write(f"{c}: {p:.3f}")
        st.progress(float(np.clip(p, 0, 1)))


if __name__ == "__main__":
    main()
