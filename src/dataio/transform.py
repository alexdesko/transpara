"""Image transforms used across datasets and scripts."""

from torchvision.models import ResNet18_Weights


def custom_transform(input_size: int = 128):
    """Standard preprocessing for grayscale images.

    Converts PIL image to tensor, resizes to a square, and normalizes to
    zero-mean, unit-ish scale around 0 using mean/std of 0.5.

    Args:
        input_size: Target image size (height=width).

    Returns:
        A ``torchvision.transforms.Compose`` callable.
    """

    # Use torchvision's pretrained ResNet18 transforms. This returns a
    # callable transform pipeline; ensure we instantiate it.
    return ResNet18_Weights.DEFAULT.transforms()
