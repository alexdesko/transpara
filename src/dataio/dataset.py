import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T

from .transform import custom_transform
from typing import Iterable, Tuple, Union
import os

class CustomDataset(Dataset):
    """Minimal dataset over precomputed (path, label) pairs.

    This simple wrapper expects an iterable of tuples ``(image_path, label)``
    where ``image_path`` is a ``pathlib.Path`` (or str) to a PNG/JPG image and
    ``label`` is an integer class index. Items are loaded with PIL and returned
    as a tensor along with the label.
    """

    def __init__(self, path_and_idx: Iterable[Tuple[Path, int]], transform = custom_transform()):
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


class CustomDataset_old(Dataset):
    """
    Generic image dataset with automatic 80/10/10 split.

    Supports two directory layouts:
    1) Single root containing class folders with images:
       <path>/{CLASS_A,CLASS_B,...}/*.{jpg,jpeg,png}
       In this case, we deterministically split per class into train/val/test
       with a 80/10/10 proportion using a fixed seed.

    2) Pre-split root with explicit split subfolders:
       <path>/{train,val,test}/{CLASS_A,CLASS_B,...}/*.{jpg,jpeg,png}

    Args:
        path: Dataset root folder.
        split: One of {"train", "val", "test"}.
        transform: Optional transform to apply to PIL images.
        seed: Random seed used for deterministic split when auto-splitting.
    """

    IMG_PATTERNS = ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG")

    def __init__(self, path: Union[str, Path], split: str, transform=None, seed: int = 42):
        base = Path(path)
        if not base.exists():
            raise FileNotFoundError(f"Dataset root {base} does not exist.")

        split = split.lower()
        assert split in {"train", "val", "test"}, "split must be one of: train, val, test"

        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        # Detect layout: pre-split or single root
        pre_split = all((base / s).exists() for s in ("train", "val", "test"))

        if pre_split:
            # Expect classes under each split directory
            split_dir = base / split
            class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
            if not class_dirs:
                raise FileNotFoundError(f"No class folders found under {split_dir}")
            class_names = sorted([p.name for p in class_dirs])
            self.class_to_idx = {name: i for i, name in enumerate(class_names)}
            self.class_names = class_names

            for class_name in class_names:
                class_path = split_dir / class_name
                for pattern in self.IMG_PATTERNS:
                    for img_file in class_path.glob(pattern):
                        self.samples.append((img_file, self.class_to_idx[class_name]))
        else:
            # Single root with class folders -> build deterministic split
            class_dirs = [p for p in base.iterdir() if p.is_dir()]
            if not class_dirs:
                raise FileNotFoundError(f"No class folders found under {base}")
            class_names = sorted([p.name for p in class_dirs])
            self.class_to_idx = {name: i for i, name in enumerate(class_names)}
            self.class_names = class_names

            rng = random.Random(seed)
            per_class_files: dict[str, list[Path]] = {}
            for class_name in class_names:
                files: list[Path] = []
                for pattern in self.IMG_PATTERNS:
                    files.extend((base / class_name).glob(pattern))
                files = sorted(files)  # deterministic order before shuffling
                rng.shuffle(files)
                per_class_files[class_name] = files

            # Create split indices 80/10/10 per class
            for class_name, files in per_class_files.items():
                n = len(files)
                n_train = int(0.8 * n)
                n_val = int(0.1 * n)
                n_test = n - n_train - n_val
                if split == "train":
                    chosen = files[:n_train]
                elif split == "val":
                    chosen = files[n_train : n_train + n_val]
                else:  # test
                    chosen = files[n_train + n_val : n_train + n_val + n_test]
                label = self.class_to_idx[class_name]
                self.samples.extend((p, label) for p in chosen)

        if not self.samples:
            raise RuntimeError(f"No images found for split={split} under {base}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        # Ensure 3 channels if a custom transform returns single-channel
        try:
            if getattr(image, "dim", None) and image.dim() == 2:
                image = image.unsqueeze(0)
            if getattr(image, "dim", None) and image.dim() == 3 and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        except Exception:
            pass
        return image, label

    def get_labels(self):
        return [label for _, label in self.samples]
