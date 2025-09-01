from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from app_utils import find_runs, load_config, load_metrics
from omegaconf import OmegaConf

# Ensure local utils take precedence over any installed package named "streamlit_app"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Training Runs", layout="wide")


def main():
    st.header("Training Runs")

    runs = find_runs()
    if not runs:
        st.warning("No runs found under outputs/ or trained_models/.")
        return

    run = st.selectbox("Select a run", options=runs, format_func=lambda p: str(p))
    cfg = load_config(run)
    st.subheader("Config")
    st.caption("Model config")
    st.json(OmegaConf.to_container(cfg, resolve=True))

    st.subheader("Metrics")
    df: pd.DataFrame = load_metrics(run)
    st.dataframe(df.tail(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(df[["train_loss", "val_loss"]])
    with col2:
        st.line_chart(df[["train_accuracy", "val_accuracy"]])


if __name__ == "__main__":
    main()
