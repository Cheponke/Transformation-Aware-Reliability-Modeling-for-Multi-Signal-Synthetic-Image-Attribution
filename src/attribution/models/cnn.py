"""Backbone CNNs (ResNet-50, EfficientNet-B0, ResNet-18 baseline) with a single-logit head."""
import torch.nn as nn
from torchvision import models

_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_model(arch: str, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    builder, weights = _BUILDERS[arch]
    model = builder(weights=weights if pretrained else None)
    head = lambda n: nn.Sequential(nn.Dropout(dropout), nn.Linear(n, 1))
    if arch.startswith("resnet"):
        model.fc = head(model.fc.in_features)
    else:
        model.classifier = head(model.classifier[-1].in_features)
    return model
