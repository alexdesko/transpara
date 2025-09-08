import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


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

    def __init__(self, num_classes=3):
        super(CustomResNet18, self).__init__()
        # Build base model
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        for p in self.model.parameters():
            p.requires_grad = False
        # Replace the final FC for desired number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        self.model.fc.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
    def freeze_layers(self):
        for p in self.model.paramters():
            p.requires_grad = False
        self.model.fc.requires_grad = True
    
    def unfreeze_layers(self):
        for p in self.model.parameters():
            p.requires_grad = True
