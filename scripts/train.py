import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.schema import AppConfig
from dataio import CustomDataset, custom_transform
from models import CustomResNet18, SimpleCNN, SimpleMLP
from trainer import Trainer
from utils.device import get_device

# @main(config_path="../configs", config_name="train", version_base=None)
# def run(cfg: DictConfig) -> None:
#     """Hydra entrypoint for training.

#     Reads configuration from ``configs/train.yaml`` (and CLI overrides),
#     prepares datasets/loaders, constructs the requested model and optimizer,
#     trains for the configured number of epochs, and writes artifacts
#     (``config.yaml``, ``model.pth``, ``metrics.csv``) into the Hydra run dir.
#     """
#     print("Configuration:\n" + OmegaConf.to_yaml(cfg))

#     # Resolve dataset root relative to project with Hydra
#     data_root = Path(to_absolute_path(cfg.data.root))
#     input_size = cfg.data.input_size

#     # Datasets
#     train_set = CustomDataset(
#         path=data_root,
#         split="train",
#         transform=custom_transform(input_size=input_size),
#     )
#     val_set = CustomDataset(
#         path=data_root,
#         split="val",
#         transform=custom_transform(input_size=input_size),
#     )

#     # Sampler (optional)
#     sampler = None
#     if cfg.training.use_weighted_sampler:
#         labels = train_set.get_labels()
#         class_sample_counts = torch.bincount(torch.tensor(labels))
#         class_weights = 1.0 / class_sample_counts.float()
#         weights = [class_weights[label] for label in labels]
#         sampler = torch.utils.data.WeightedRandomSampler(
#             weights=weights,
#             num_samples=len(train_set),
#             replacement=True,
#         )

#     # Dataloaders
#     loader = DataLoader(
#         train_set,
#         batch_size=cfg.training.batch_size,
#         shuffle=(sampler is None),
#         sampler=sampler,
#         num_workers=cfg.dataloader.num_workers,
#         pin_memory=cfg.dataloader.pin_memory,
#         persistent_workers=cfg.dataloader.persistent_workers,
#     )
#     val_loader = DataLoader(
#         val_set,
#         batch_size=cfg.training.batch_size,
#         shuffle=False,
#         num_workers=cfg.dataloader.num_workers,
#         pin_memory=cfg.dataloader.pin_memory,
#         persistent_workers=cfg.dataloader.persistent_workers,
#     )

#     # Models
#     if cfg.model.name == "CNN":
#         model = SimpleCNN(input_size=input_size, num_classes=cfg.model.num_classes)
#     elif cfg.model.name == "MLP":
#         model = SimpleMLP(
#             input_size=input_size * input_size, hidden_size=cfg.model.hidden_size, num_classes=cfg.model.num_classes
#         )
#     elif cfg.model.name == "ResNet18":
#         model = CustomResNet18(num_classes=cfg.model.num_classes, in_channels=1, weights=None)
#     else:
#         raise ValueError(f"Unknown model type: {cfg.model.name}")

#     # Optimizer
#     if cfg.optimizer.name == "Adam":
#         optimizer = Adam(model.parameters(), lr=cfg.optimizer.lr)
#     else:
#         raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")

#     # Criterion
#     if cfg.criterion.name == "CrossEntropyLoss":
#         criterion = CrossEntropyLoss()
#     elif cfg.criterion.name == "WeightedCrossEntropyLoss":
#         labels = train_set.get_labels()
#         class_counts = torch.bincount(torch.tensor(labels))
#         total_samples = len(labels)
#         class_weights = total_samples / (len(class_counts) * class_counts.float())
#         print(f"Class weights: {class_weights}")
#         criterion = CrossEntropyLoss(weight=class_weights)
#     else:
#         raise ValueError(f"Unknown criterion: {cfg.criterion.name}")

#     # Device
#     device = get_device()
#     model = model.to(device)

#     # Trainer
#     trainer = Trainer(
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         num_epochs=cfg.training.num_epochs,
#         device=device,
#         use_amp=cfg.training.use_amp,
#     )
#     # Configure metric to monitor for best checkpoint
#     if hasattr(cfg.training, "metric_to_monitor") and hasattr(cfg.training, "mode"):
#         try:
#             trainer.configure_monitor(cfg.training.metric_to_monitor, cfg.training.mode)
#         except Exception:
#             pass

#     # Reproducibility
#     try:
#         torch.manual_seed(cfg.training.seed)
#         random.seed(cfg.training.seed)
#         np.random.seed(cfg.training.seed)
#         if device.type == "cuda":
#             torch.cuda.manual_seed_all(cfg.training.seed)
#     except Exception:
#         pass

#     # Train
#     trainer.train(loader, val_loader)

#     # Save artifacts into Hydra run dir (cwd)
#     out_dir = Path(".")
#     with open(out_dir / "config.yaml", "w") as f:
#         f.write(OmegaConf.to_yaml(cfg))
#     trainer.save_model(out_dir / "model.pth")
#     trainer.save_best_model(out_dir / "best_model.pth")
#     trainer.save_metrics(out_dir / "metrics.csv")


@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(cfg.model.name)


if __name__ == "__main__":
    main()
