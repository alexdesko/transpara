from pathlib import Path

import pandas as pd
import torch
from rich.progress import Progress

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class Trainer:
    def __init__(self, model, criterion, optimizer, num_epochs=100, **kwargs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def epoch(self, train_loader):
        print("Training for one epoch...")

        running_loss = 0.0
        running_accuracy = 0.0
        self.model.train()

        with Progress() as progress:
            task = progress.add_task("[red]Training...", total=len(train_loader))

            for inputs, labels in train_loader:
                outputs = self.model(inputs.to(DEVICE)).cpu()
                loss = self.criterion(outputs, labels)
                accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
                running_accuracy += accuracy * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                progress.update(
                    task,
                    advance=1,
                    description=f"[red]Training... Loss: {loss.item():.4f}",
                )

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = running_accuracy / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)

    def validation(self, val_loader):
        print("Validating for one epoch...")

        running_loss, running_accuracy = 0.0, 0.0
        self.model.eval()

        with torch.no_grad():
            with Progress() as progress:
                task = progress.add_task("[green]Validating...", total=len(val_loader))

                for inputs, labels in val_loader:
                    outputs = self.model(inputs.to(device=DEVICE)).cpu()
                    accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    running_accuracy += accuracy * inputs.size(0)
                    progress.update(
                        task,
                        advance=1,
                        description=f"[green]Validating... Loss: {loss.item():.4f}",
                    )

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = running_accuracy / len(val_loader.dataset)
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)
        print(f"Validation Loss: {epoch_loss:.4f}")

    def train(self, train_loader, val_loader):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.epoch(train_loader)
            self.validation(val_loader)

    def save_model(self, path):
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def save_metrics(self, path):
        # Ensure the directory exists
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
        self.model.load_state_dict(torch.load(path))
