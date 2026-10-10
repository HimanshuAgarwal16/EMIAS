import torch.nn as nn
from torchvision.models import resnet18


def create_ecg_model(num_classes=4):

    model = resnet18(weights=None)

    # Replace the final classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model