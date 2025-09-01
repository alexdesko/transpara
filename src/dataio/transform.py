"""Image transforms used across datasets and scripts."""

from torchvision import transforms


def custom_transform(input_size: int = 128) -> transforms.Compose:
    """Standard preprocessing for grayscale images.

    Converts PIL image to tensor, resizes to a square, and normalizes to
    zero-mean, unit-ish scale around 0 using mean/std of 0.5.

    Args:
        input_size: Target image size (height=width).

    Returns:
        A ``torchvision.transforms.Compose`` callable.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
