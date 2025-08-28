from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

HOME = "/Users/alexdesko/Documents/code/transpara"


class CustomDataset(Dataset):
    """
    Custom Dataset for loading images from a directory structure.
    """

    def __init__(self, path, split, transform=None):
        # check if path exists
        self.path = Path(HOME) / path / split
        if not self.path.exists():
            raise FileNotFoundError(f"Directory {self.path} does not exist.")

        self.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}

        self.samples = []

        self.transform = transform

        for class_ in ["NORMAL", "PNEUMONIA"]:
            class_path = self.path / class_
            if not class_path.exists():
                raise FileNotFoundError(
                    f"Directory {class_path} does not exist. Please ensure the dataset is correctly placed."
                )
            for img_file in class_path.glob("*.jpeg"):
                self.samples.append((img_file, self.class_to_idx[class_]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # convert image in BW
        image = Image.open(img_path).convert("L")
        # Load image
        if self.transform:
            image = self.transform(image)
        else:
            image = pil_to_tensor(image).float() / 255.0  # Normalize to [0, 1]
        return image, label

    def get_labels(self):
        return [label for _, label in self.samples]
