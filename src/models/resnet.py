from typing import List

import torch.nn as nn
import torchvision


def build_model(cfg, num_classes: int) -> nn.Module:
    """ResNet18 with a small classifier head."""
    mcfg = cfg.get("model", {})
    backbone_name = mcfg.get("backbone", "resnet18")
    pretrained = bool(mcfg.get("pretrained", True))
    freeze = bool(mcfg.get("freeze_backbone", True))
    dropout = float(mcfg.get("dropout", 0.2))

    if backbone_name != "resnet18":
        raise ValueError("This minimal demo implements resnet18 only.")

    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = ("fc" in name)

    return model


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


__all__ = ["build_model", "get_trainable_params"]
