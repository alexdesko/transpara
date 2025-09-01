from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ModelName(str, Enum):
    CNN = "CNN"
    MLP = "MLP"
    ResNet18 = "ResNet18"


class OptimizerName(str, Enum):
    Adam = "Adam"


class CriterionName(str, Enum):
    CrossEntropyLoss = "CrossEntropyLoss"
    WeightedCrossEntropyLoss = "WeightedCrossEntropyLoss"


@dataclass
class ModelConfig:
    name: ModelName = ModelName.CNN
    num_classes: int = 2
    # For MLP only
    hidden_size: int = 512


@dataclass
class DataConfig:
    root: str = "chest_xray"
    input_size: int = 256


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    use_weighted_sampler: bool = True
    use_amp: bool = False
    seed: int = 42
    metric_to_monitor: Literal["val_loss", "val_accuracy"] = "val_accuracy"
    mode: Literal["min", "max"] = "max"


@dataclass
class DataloaderConfig:
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class OptimizerConfig:
    name: OptimizerName = OptimizerName.Adam
    lr: float = 1e-3


@dataclass
class CriterionConfig:
    name: CriterionName = CriterionName.CrossEntropyLoss


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    experiment_name: str = "${model.name}_${criterion.name}"

