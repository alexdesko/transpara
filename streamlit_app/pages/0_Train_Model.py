from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import yaml
from enum import Enum
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure local utils take precedence over any installed package named "streamlit_app"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make src/ importable (dataio, models, utils)
REPO_ROOT = ROOT.parent
SRC_DIR = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from app_utils import build_model  # noqa: E402 (ensures sys.path injection as well)

from configs.schema import AppConfig  # noqa: E402
from dataio import CustomDataset, custom_transform  # noqa: E402
from trainer import Trainer  # noqa: E402
from utils.device import get_device  # noqa: E402

st.set_page_config(page_title="Train Model", layout="wide")


def _default_cfg() -> AppConfig:
    return AppConfig()


def _make_run_dir(cfg: AppConfig) -> Path:
    # Mirror Hydra's outputs path scheme
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H-%M-%S")
    exp = f"{cfg.model.name}_{cfg.criterion.name}"
    run_dir = REPO_ROOT / "outputs" / date / f"{time}_{exp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_config(run_dir: Path, cfg: AppConfig) -> None:
    # Persist as YAML alongside artifacts, coercing Enums to primitives
    def _coerce(obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ _coerce(v) for v in obj ]
        return obj

    data = _coerce(asdict(cfg))
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _ui_config() -> AppConfig:
    cfg = _default_cfg()

    st.subheader("Configuration")
    col_data, col_model = st.columns(2)
    with col_data:
        st.caption("Data")
        cfg.data.root = st.text_input("Dataset root", value=cfg.data.root, help="Path to dataset root folder")
        cfg.data.input_size = int(
            st.number_input("Input size (H=W)", min_value=64, max_value=1024, value=int(cfg.data.input_size), step=32)
        )

        st.caption("Dataloader")
        cfg.dataloader.num_workers = int(
            st.number_input("Num workers", min_value=0, max_value=16, value=int(cfg.dataloader.num_workers), step=1)
        )
        cfg.dataloader.pin_memory = bool(st.checkbox("Pin memory", value=bool(cfg.dataloader.pin_memory)))
        cfg.dataloader.persistent_workers = bool(
            st.checkbox("Persistent workers", value=bool(cfg.dataloader.persistent_workers))
        )

    with col_model:
        st.caption("Model")
        model_name = st.selectbox("Model", options=["ResNet18", "CNN", "MLP"], index=0)
        cfg.model.name = model_name  # type: ignore[assignment]
        cfg.model.num_classes = int(
            st.number_input("Num classes", min_value=2, max_value=10, value=cfg.model.num_classes)
        )
        if model_name == "MLP":
            cfg.model.hidden_size = int(
                st.number_input("MLP hidden size", min_value=64, max_value=4096, value=cfg.model.hidden_size, step=64)
            )

        st.caption("Optimizer & Criterion")
        cfg.optimizer.lr = float(
            st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, step=1e-4, format="%.6f")
        )
        crit = st.selectbox("Criterion", options=["CrossEntropyLoss", "WeightedCrossEntropyLoss"], index=0)
        cfg.criterion.name = crit  # type: ignore[assignment]

    st.caption("Training")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        cfg.training.batch_size = int(
            st.number_input("Batch size", min_value=1, max_value=512, value=cfg.training.batch_size, step=1)
        )
        cfg.training.num_epochs = int(
            st.number_input("Epochs", min_value=1, max_value=1000, value=cfg.training.num_epochs, step=1)
        )
        cfg.training.seed = int(st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=cfg.training.seed))
    with col_t2:
        cfg.training.use_weighted_sampler = bool(
            st.checkbox("Weighted sampler", value=bool(cfg.training.use_weighted_sampler))
        )
        cfg.training.use_amp = bool(st.checkbox("Use AMP (CUDA)", value=bool(cfg.training.use_amp)))
    with col_t3:
        metric = st.selectbox("Monitor metric", options=["val_accuracy", "val_loss"], index=0)
        cfg.training.metric_to_monitor = metric  # type: ignore[assignment]
        mode_default = "max" if metric == "val_accuracy" else "min"
        cfg.training.mode = st.selectbox("Mode", options=["max", "min"], index=["max", "min"].index(mode_default))

    return cfg


def _prepare_data(cfg: AppConfig):
    data_root = Path(cfg.data.root)
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()

    transform = custom_transform(input_size=cfg.data.input_size)
    train_set = CustomDataset(path=data_root, split="train", transform=transform)
    val_set = CustomDataset(path=data_root, split="val", transform=transform)

    sampler = None
    if cfg.training.use_weighted_sampler:
        labels = train_set.get_labels()
        counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / counts.float().clamp_min(1)
        weights = [class_weights[i] for i in labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_set), replacement=True)

    # persistent_workers requires num_workers > 0
    persistent = cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


def _prepare_model_and_optim(cfg: AppConfig):
    model = build_model(cfg.model.name, cfg.data.input_size, cfg.model.num_classes)
    optimizer = Adam(model.parameters(), lr=cfg.optimizer.lr)

    if cfg.criterion.name == "CrossEntropyLoss":
        criterion = CrossEntropyLoss()
    else:
        # Weighted CE will be created later once we have label counts; placeholder here
        criterion = None  # type: ignore[assignment]

    return model, optimizer, criterion


def _train_loop(cfg: AppConfig) -> Path:
    device = get_device()
    st.info(f"Using device: {device}")

    run_dir = _make_run_dir(cfg)
    _save_config(run_dir, cfg)

    # Data
    try:
        train_loader, val_loader = _prepare_data(cfg)
    except Exception as e:
        st.error(f"Failed to prepare data: {e}")
        raise

    # Model/optim/criterion
    model, optimizer, criterion = _prepare_model_and_optim(cfg)

    # If weighted criterion requested, compute class weights from training set
    if criterion is None:
        labels = getattr(train_loader.dataset, "get_labels", lambda: [])()
        counts = torch.bincount(torch.tensor(labels))
        total = max(1, int(sum(counts)))
        weights = total / (len(counts) * counts.float().clamp_min(1))
        st.caption(f"Class weights: {weights.tolist()}")
        criterion = CrossEntropyLoss(weight=weights)

    model = model.to(device)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.training.num_epochs,
        device=device,
        use_amp=cfg.training.use_amp,
    )
    trainer.configure_monitor(cfg.training.metric_to_monitor, cfg.training.mode)

    # Reproducibility
    try:
        torch.manual_seed(cfg.training.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.training.seed)
    except Exception:
        pass

    # Live UI elements
    progress = st.progress(0)
    status = st.empty()
    chart_loss = st.empty()
    chart_acc = st.empty()

    metrics_df = pd.DataFrame(columns=["train_loss", "val_loss", "train_accuracy", "val_accuracy"])

    for epoch in range(cfg.training.num_epochs):
        status.write(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        trainer.epoch(train_loader)
        trainer.validation(val_loader)

        # Update metrics table
        row = {
            "train_loss": trainer.train_losses[-1],
            "val_loss": trainer.val_losses[-1],
            "train_accuracy": trainer.train_accuracies[-1],
            "val_accuracy": trainer.val_accuracies[-1],
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)

        # Refresh charts
        with chart_loss.container():
            st.line_chart(metrics_df[["train_loss", "val_loss"]])
        with chart_acc.container():
            st.line_chart(metrics_df[["train_accuracy", "val_accuracy"]])

        progress.progress(int(100 * (epoch + 1) / cfg.training.num_epochs))

    # Save artifacts
    trainer.save_model(run_dir / "model.pth")
    trainer.save_best_model(run_dir / "best_model.pth")
    trainer.save_metrics(run_dir / "metrics.csv")

    st.success("Training complete.")
    st.caption(f"Artifacts saved to: {run_dir}")
    return run_dir


def main():
    st.header("Train Model")

    cfg = _ui_config()

    with st.expander("Preview config (YAML)", expanded=False):
        # Show the same YAML we will persist on disk
        def _coerce(obj):
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [ _coerce(v) for v in obj ]
            return obj

        st.code(yaml.safe_dump(_coerce(asdict(cfg)), sort_keys=False), language="yaml")

    if st.button("Start Training", type="primary"):
        try:
            _train_loop(cfg)
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
