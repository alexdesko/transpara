from pathlib import Path

import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from dataio import CustomDataset, custom_transform
from models import SimpleCNN, SimpleMLP
from trainer import Trainer

config = yaml.safe_load(open(Path(__file__).parent.parent / "configs/train.yaml"))


print(f"Configuration: {config}")
saving_path = Path("trained_models/resnet_weighted_loss")
saving_path.mkdir(parents=True, exist_ok=True)
# write the config to the saving path
with open(saving_path / "config.yaml", "w") as f:
    yaml.dump(config, f)


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Load a data loader here
input_size = config["input_size"]
batch_size = config["batch_size"]


dataset = CustomDataset(
    path="chest_xray",
    split="train",
    transform=custom_transform(input_size=input_size),
)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_dataset = CustomDataset(
    path="chest_xray",
    split="val",
    transform=custom_transform(input_size=input_size),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# Prepare the model here
match config["model"]:
    case "CNN":
        model = SimpleCNN(
            input_size=input_size,
            num_classes=2,
        )
    case "MLP":
        model = SimpleMLP(
            input_size=input_size * input_size,
            hidden_size=512,
            num_classes=2,
        )
    case "ResNet18":
        model = resnet18(weights=None)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)

# Prepare the optimizer and the critetion here
match config["optimizer"]:
    case "Adam":
        optimizer = Adam(
            model.parameters(),
            lr=config["learning_rate"],
        )


match config["criterion"]:
    case "CrossEntropyLoss":
        criterion = CrossEntropyLoss()
    case "WeightedCrossEntropyLoss":
        labels = dataset.get_labels()
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        print(f"Class weights: {class_weights}")
        criterion = CrossEntropyLoss(weight=class_weights)


trainer = Trainer(
    model=model.to(device=DEVICE),
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=config["num_epochs"],
)


## Begin training here
trainer.train(loader, val_loader)

# save the model and the metrics
trainer.save_model(saving_path / "model.pth")
trainer.save_metrics(saving_path / "metrics.csv")
