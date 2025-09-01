"""Convolutional model definitions.

This module contains a minimal convolutional baseline used for grayscale
chest X‑ray classification tasks. The architecture is intentionally simple
to serve as a baseline for experiments.
"""

import torch
import torch.nn as nn

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class SimpleCNN(nn.Module):
    """A simple CNN baseline for 1‑channel images.

    Note: The classifier's first linear layer currently assumes a flattened
    size of ``input_size * input_size``. This is a minimal baseline and does
    not compute the post‑conv spatial size; adjust as needed if you modify
    the feature extractor.

    Args:
        input_size: Input image height/width in pixels (images are square).
        num_classes: Number of output classes.
    """

    def __init__(self, input_size, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size * input_size, 128),  # Adjust based on input image size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Compute logits for a batch.

        Args:
            x: Tensor of shape ``(N, 1, H, W)``.

        Returns:
            Tensor of shape ``(N, num_classes)`` with unnormalized logits.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
