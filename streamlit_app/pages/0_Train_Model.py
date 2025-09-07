from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

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

from app_utils import build_model  # noqa: E402
from dataio.utils import create_datasets  # noqa: E402
from trainer import Trainer  # noqa: E402
from utils.device import get_device  # noqa: E402

st.set_page_config(page_title="Train Model", layout="wide")


def _make_run_dir(model_name: str, criterion_name: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H-%M-%S")
    exp = f"{model_name}_{criterion_name}"
    run_dir = REPO_ROOT / "outputs" / date / f"{time}_{exp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _ui_config() -> dict:
    st.subheader("Configuration")
    col_left, col_right = st.columns(2)

    with col_left:
        data_root = st.text_input("Dataset root", value="dataset")
        input_size = int(st.number_input("Input size", min_value=64, max_value=1024, value=256, step=32))
        batch_size = int(st.number_input("Batch size", min_value=1, max_value=512, value=32, step=1))
        num_epochs = int(st.number_input("Epochs", min_value=1, max_value=1000, value=50, step=1))

    with col_right:
        lr = float(st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, step=1e-4, format="%.6f"))
        num_classes = int(st.number_input("Num classes", min_value=2, max_value=10, value=3, step=1))

    cfg = {
        "model": {"name": "ResNet18", "num_classes": num_classes},
        "data": {"root": data_root, "input_size": input_size},
        "training": {"batch_size": batch_size, "num_epochs": num_epochs, "seed": 42, "metric_to_monitor": "val_accuracy", "mode": "max"},
        "optimizer": {"lr": lr},
        "criterion": {"name": "CrossEntropyLoss"},
    }
    return cfg


def _prepare_data(cfg: dict):
    data_root = Path(cfg["data"]["root"]).resolve()
    train_set, val_set, test_set = create_datasets(path=data_root)

    bs = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
    # Static class names used by dataset split
    class_names = ["NORMAL", "PNEUMONIA", "COVID"]
    return train_loader, val_loader, test_loader, class_names


def _prepare_model_and_optim(cfg: dict):
    model = build_model(cfg["model"]["name"], cfg["data"]["input_size"], cfg["model"]["num_classes"])
    optimizer = Adam(model.parameters(), lr=float(cfg["optimizer"]["lr"]))
    criterion = CrossEntropyLoss()
    return model, optimizer, criterion


def _train_loop(cfg: dict) -> Path:
    device = get_device()
    st.info(f"Using device: {device}")

    run_dir = _make_run_dir(cfg["model"]["name"], cfg["criterion"]["name"])  # Hydra-like outputs path

    # Data
    try:
        train_loader, val_loader, test_loader, class_names = _prepare_data(cfg)
    except Exception as e:
        st.error(f"Failed to prepare data: {e}")
        raise

    # Model/optim/criterion
    model, optimizer, criterion = _prepare_model_and_optim(cfg)

    model = model.to(device)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=int(cfg["training"]["num_epochs"]),
        device=device,
    )
    trainer.configure_monitor(str(cfg["training"]["metric_to_monitor"]), str(cfg["training"]["mode"]))

    # Reproducibility
    try:
        torch.manual_seed(int(cfg["training"].get("seed", 42)))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(cfg["training"].get("seed", 42)))
    except Exception:
        pass

    # Live UI elements
    progress = st.progress(0)
    status = st.empty()
    chart_loss = st.empty()
    chart_acc = st.empty()

    metrics_df = pd.DataFrame(columns=["train_loss", "val_loss", "train_accuracy", "val_accuracy"])

    for epoch in range(int(cfg["training"]["num_epochs"])):
        status.write(f"Epoch {epoch + 1}/{cfg["training"]["num_epochs"]}")
        trainer.epoch(train_loader)
        trainer.validation(val_loader)

        row = {
            "train_loss": trainer.train_losses[-1],
            "val_loss": trainer.val_losses[-1],
            "train_accuracy": trainer.train_accuracies[-1],
            "val_accuracy": trainer.val_accuracies[-1],
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)

        with chart_loss.container():
            st.line_chart(metrics_df[["train_loss", "val_loss"]])
        with chart_acc.container():
            st.line_chart(metrics_df[["train_accuracy", "val_accuracy"]])

        progress.progress(int(100 * (epoch + 1) / int(cfg["training"]["num_epochs"])))

    # Evaluate on test set
    with torch.no_grad():
        running_loss, running_correct, total = 0.0, 0, 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(1)
            running_correct += (preds == y).sum().item()
            running_loss += float(loss.item()) * x.size(0)
            total += x.size(0)
        test_loss = running_loss / max(1, total)
        test_acc = running_correct / max(1, total)
        st.metric("Test accuracy", f"{test_acc:.3f}")

    # Save artifacts and minimal config for other pages
    cfg_to_save = {
        "model": cfg["model"],
        "data": cfg["data"],
        "optimizer": cfg["optimizer"],
        "training": {k: cfg["training"][k] for k in ("batch_size", "num_epochs", "seed")},
        "class_names": class_names,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_to_save, f, sort_keys=False)

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
        st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")

    if st.button("Start Training", type="primary"):
        try:
            _train_loop(cfg)
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
