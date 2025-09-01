import torch
import torch.nn as nn
from torchvision.models import resnet18 as tv_resnet18


class CustomResNet18(nn.Module):
    """ResNet18 wrapper with first/last layers adapted.

    This wraps ``torchvision.models.resnet18`` and:
    - Replaces the first conv to support arbitrary ``in_channels`` (default 1).
    - Replaces the final fully connected layer to match ``num_classes``.

    If ``weights`` are provided (e.g., torchvision weight enums) and
    ``in_channels == 1``, the original RGB conv1 weights are averaged across
    channels to initialize the 1‑channel conv for smoother transfer.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (1 for grayscale).
        weights: Optional torchvision ResNet18 weights for initialization.

    Returns:
        A module whose forward returns logits of shape ``(N, num_classes)``.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 1, weights=None):
        super().__init__()
        # Build base model
        self.model = tv_resnet18(weights=weights)

        # Adapt first conv for grayscale or other channel counts
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # If using pretrained weights and going from 3->1 channels, average weights
        if (
            hasattr(old_conv, "weight")
            and old_conv.weight.shape[1] == 3
            and in_channels == 1
            and weights is not None
        ):
            with torch.no_grad():
                w = old_conv.weight.detach().clone()
                # Average across RGB input channel dimension
                w = w.mean(dim=1, keepdim=True)
                self.model.conv1.weight.copy_(w)

        # Replace the final FC for desired number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(
            in_features=in_features, out_features=num_classes, bias=True
        )

    def forward(self, x):
        return self.model(x)
