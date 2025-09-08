from torchvision.models import ResNet18_Weights

def custom_transform():
    return ResNet18_Weights.DEFAULT.transforms()