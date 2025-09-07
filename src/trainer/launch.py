from __future__ import annotations

"""Hydra launcher for training with a single config file.

This entrypoint prepares train/val/test splits (80/10/10), constructs a
ResNet18 model, and runs training with artifacts saved under Hydra's
``outputs/`` directory. It is callable from the terminal and can be reused
from Streamlit.
"""

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler


from models import CustomResNet18
from trainer import Trainer
from dataio import create_datasets

import os
DEBUG = os.environ["DEBUG"]

def _prepare_loaders(cfg: DictConfig):
    data_root = Path(cfg.data.root)
    if not data_root.is_absolute():
        data_root = Path(hydra.utils.get_original_cwd()) / data_root


    train_set, val_set, test_set = create_datasets(path = data_root)

    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False)
    test_loader = DataLoader(test_set, shuffle=False)

    return train_loader, val_loader, test_loader


def _build_model(cfg: DictConfig):
    # Only ResNet is kept
    num_classes = int(cfg.model.num_classes)
    model = CustomResNet18(num_classes=num_classes)
    return model


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module):
    lr = float(cfg.optimizer.get("lr", 1e-3))
    wd = float(cfg.optimizer.get("lr", 1e-5))
    return Adam(model.parameters(), lr=lr, weight_decay=wd)


def _build_criterion(cfg: DictConfig, train_labels: list[int] | None = None):
    name = str(cfg.criterion.name)
    if name == "CrossEntropyLoss" or train_labels is None:
        return CrossEntropyLoss()
    # Weighted CE
    counts = torch.bincount(torch.tensor(train_labels))
    total = max(1, int(sum(counts)))
    weights = total / (len(counts) * counts.float().clamp_min(1))
    return CrossEntropyLoss(weight=weights)


def run(cfg: DictConfig) -> None:
    print("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Data
    train_loader, val_loader, test_loader = _prepare_loaders(cfg)
    print(
        f"Datasets -> train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}"
    )

    # Model/optim/criterion
    model = _build_model(cfg)
    optimizer = _build_optimizer(cfg, model)
    criterion = _build_criterion(cfg, getattr(train_loader.dataset, "get_labels", lambda: [])())

    # Device
    device = torch.device("cpu")
    model = model.to(device)

    # Reproducibility
    try:
        seed = int(cfg.training.get("seed", 42))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    

    num_epochs = 10 if DEBUG else int(cfg.training.num_epochs) 
    # Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )
    trainer.configure_monitor(str(cfg.training.get("metric_to_monitor", "val_accuracy")), str(cfg.training.get("mode", "max")))
    trainer.train(train_loader, val_loader)

    # Save artifacts in Hydra run dir (cwd is the run directory)
    out_dir = Path('./temp_DEBUG')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Augment config with class names for downstream apps
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(out_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg_dict))
    trainer.save_model(out_dir / "model.pth")
    trainer.save_best_model(out_dir / "best_model.pth")
    trainer.save_metrics(out_dir / "metrics.csv")

CONFIG_DIR = str((Path(__file__).resolve().parents[2] / "configs").resolve())

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
