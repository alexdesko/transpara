"""Image transforms used across datasets and scripts."""

from torchvision.models import ResNet18_Weights


def custom_transform():
    """Standard preprocessing for grayscale images.

    Converts PIL image to tensor, resizes to a square, and normalizes to
    zero-mean, unit-ish scale around 0 using mean/std of 0.5.

    Args:
        input_size: Target image size (height=width).

    Returns:
        A ``torchvision.transforms.Compose`` callable.
    """
    return ResNet18_Weights.DEFAULT.transforms()
