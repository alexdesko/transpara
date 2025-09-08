import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights

from typing import Iterable, Tuple
import os

class CustomDataset(Dataset):
    """Minimal dataset over precomputed (path, label) pairs.

    This simple wrapper expects an iterable of tuples ``(image_path, label)``
    where ``image_path`` is a ``pathlib.Path`` (or str) to a PNG/JPG image and
    ``label`` is an integer class index. Items are loaded with PIL and returned
    as a tensor along with the label.
    """

    def __init__(self, path_and_idx: Iterable[Tuple[Path, int]], transform = ResNet18_Weights.DEFAULT.transforms()):
        self.samples: list[Tuple[Path, int]] = [(Path(p), int(y)) for p, y in path_and_idx]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        # Minimal default without NumPy: PIL -> Tensor in [0,1]
        if self.transform:
            image = self.transform(image)
        else:
            image = pil_to_tensor(image).float().div(255)
        return image, label