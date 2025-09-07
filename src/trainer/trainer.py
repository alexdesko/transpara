"""Training utilities.

This module provides a lightweight Trainer abstraction that handles the
training/validation loop, optional mixed precision on CUDA, progress display,
and saving/loading of model weights and per-epoch metrics.
"""

from pathlib import Path

import pandas as pd
import torch
from rich.progress import Progress

from typing import Optional

class Trainer:
    """Simple supervised training loop with optional AMP.

    Args:
        model: Torch module to train.
        criterion: Loss function mapping ``(logits, labels) -> loss``.
        optimizer: Optimizer updating ``model.parameters()``.
        num_epochs: Number of epochs to train.
        device: Target device for training (cpu/cuda/mps).
        use_amp: Enable CUDA automatic mixed precision (AMP) if available.

    Attributes:
        train_losses: List of per-epoch training losses.
        val_losses: List of per-epoch validation losses.
        train_accuracies: List of per-epoch training accuracies.
        val_accuracies: List of per-epoch validation accuracies.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        num_epochs: int = 100,
        device: Optional[torch.device] = None
        #device: torch.device | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device or torch.device("cpu")

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []

        # Best checkpoint tracking (val_loss or val_accuracy)
        self.monitor: str = "val_loss"
        self.mode: str = "min"
        self._best_value: float | None = None
        self._best_state: dict | None = None

    def epoch(self, train_loader):
        """Run one training epoch over ``train_loader``.

        Accumulates accuracy and loss over the entire dataset using the
        configured device and AMP settings.
        """
        running_loss = 0.0
        running_correct = 0
        total = 0

        self.model.train()

        with Progress() as progress:
            task = progress.add_task("[red]Training...", total=len(train_loader))
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

                progress.update(
                    task,
                    advance=1,
                    description=f"[red]Training... Loss: {loss.item():.4f}",
                )

        epoch_loss = running_loss / total
        epoch_accuracy = running_correct / total
        print(f"Training Loss: {epoch_loss:.4f}")

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)

    def validation(self, val_loader):
        """Run one validation epoch over ``val_loader`` without gradients."""

        running_loss = 0.0
        running_correct = 0
        total = 0
        self.model.eval()

        with torch.no_grad():
            with Progress() as progress:
                task = progress.add_task("[green]Validating...", total=len(val_loader))

                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    running_correct += (preds == labels).sum().item()
                    running_loss += loss.item() * inputs.size(0)
                    total += inputs.size(0)
                    progress.update(
                        task,
                        advance=1,
                        description=f"[green]Validating... Loss: {loss.item():.4f}",
                    )

        epoch_loss = running_loss / total
        epoch_accuracy = running_correct / total
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)
        print(f"Validation Loss: {epoch_loss:.4f}")

        # Update best checkpoint
        current = epoch_loss if self.monitor == "val_loss" else epoch_accuracy
        better = (
            (self._best_value is None)
            or (self.mode == "min" and current < self._best_value)
            or (self.mode == "max" and current > self._best_value)
        )
        if better:
            self._best_value = current
            # clone to CPU for safe serialization
            self._best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def train(self, train_loader, val_loader):
        """Train for ``num_epochs``, alternating train and validation."""
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.epoch(train_loader)
            self.validation(val_loader)

    def save_model(self, path):
        """Save model weights to ``path`` (creates parent dirs)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def save_metrics(self, path):
        """Save metrics CSV with train/val loss and accuracy to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "train_loss": self.train_losses,
                "val_loss": self.val_losses,
                "train_accuracy": self.train_accuracies,
                "val_accuracy": self.val_accuracies,
            }
        )
        df.to_csv(path, index=False)

    def load_model(self, path):
        """Load model weights from ``path`` onto CPU (caller can move to device)."""
        self.model.load_state_dict(torch.load(path, map_location="cpu"))

    def configure_monitor(self, monitor: str, mode: str):
        """Set which validation metric to monitor and its direction."""
        assert monitor in {"val_loss", "val_accuracy"}
        assert mode in {"min", "max"}
        self.monitor = monitor
        self.mode = mode

    def save_best_model(self, path):
        """Save the best checkpoint observed during training, if any."""
        if self._best_state is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._best_state, path)
