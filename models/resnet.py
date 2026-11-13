"""ResNet model definitions for CIFAR-10."""

from torch import nn
from torchvision.models import resnet18


def create_model(num_classes: int = 10) -> nn.Module:
    """Create a randomly initialized ResNet-18 adapted for 32x32 images."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
