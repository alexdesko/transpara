import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class CustomDataset(Dataset):
    """
    Custom Dataset for loading images from a directory structure:
    <path>/<split>/{NORMAL,PNEUMONIA}/*.jpg|*.jpeg|*.png
    """

    def __init__(self, path: str | Path, split: str, transform=None, sampling=None):
        # Resolve path without hardcoded HOME; allow absolute or relative
        base = Path(path)
        self.path = base / split
        if not self.path.exists():
            raise FileNotFoundError(f"Directory {self.path} does not exist.")

        self.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform

        for class_ in ["NORMAL", "PNEUMONIA"]:
            class_path = self.path / class_
            if not class_path.exists():
                raise FileNotFoundError(
                    f"Directory {class_path} does not exist. Please ensure the dataset is correctly placed."
                )
            for pattern in ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
                for img_file in class_path.glob(pattern):
                    self.samples.append((img_file, self.class_to_idx[class_]))

        if sampling == "under":
            # Undersample the majority class to match the minority count
            class0 = [s for s in self.samples if s[1] == 0]
            class1 = [s for s in self.samples if s[1] == 1]
            if class0 and class1:
                target = min(len(class0), len(class1))
                rng = random.Random(42)  # deterministic undersampling
                if len(class0) > target:
                    class0 = rng.sample(class0, target)
                if len(class1) > target:
                    class1 = rng.sample(class1, target)
                self.samples = class0 + class1
                rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Load as RGB to match torchvision ResNet transforms (3-channel)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = pil_to_tensor(image).float() / 255.0  # Normalize to [0, 1]

        # Ensure the output has 3 channels by duplicating if single-channel
        try:
            if getattr(image, "dim", None) and image.dim() == 2:
                image = image.unsqueeze(0)
            if getattr(image, "dim", None) and image.dim() == 3 and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        except Exception:
            # Keep minimal behavior; if non-tensor, leave unchanged
            pass
        return image, label

    def get_labels(self):
        return [label for _, label in self.samples]
