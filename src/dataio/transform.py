from torchvision import transforms


def custom_transform(input_size=128):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
