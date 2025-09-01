"""MLP model definitions.

Provides a simple multilayer perceptron baseline that operates on flattened
grayscale images. Intended as a lightweight benchmark alongside CNN/ResNet
architectures.
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


class SimpleMLP(nn.Module):
    """A simple MLP baseline for flattened 1‑channel images.

    Args:
        input_size: Flattened input size (H*W) expected by the first layer.
        hidden_size: Hidden layer width.
        num_classes: Number of output classes.
    """

    def __init__(self, input_size=224 * 224, hidden_size=512, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """Compute logits for a batch.

        Args:
            x: Tensor of shape ``(N, 1, H, W)`` or already flattened to
               ``(N, input_size)``; input is flattened internally.

        Returns:
            Tensor of shape ``(N, num_classes)`` with unnormalized logits.
        """
        x = self.classifier(x)
        return x
