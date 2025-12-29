from typing import Dict

import torchvision.transforms as T
from PIL import Image


def build_transforms(cfg: Dict, train: bool) -> T.Compose:
    """Build preprocessing pipeline with optional augmentation strength."""
    img_size = int(cfg["data"]["image_size"])
    aug = cfg.get("aug", {})
    strength = aug.get("aug_strength", "light").lower()

    base = [
        # MedMNIST already yields PIL Images; keep them as-is and convert tensors/arrays only if needed
        T.Lambda(lambda img: img if isinstance(img, Image.Image) else T.ToPILImage()(img)),
        T.Resize((img_size, img_size)),
    ]

    if train:
        if strength == "light":
            rot = aug.get("rotation_deg", 10)
            jitter = aug.get("color_jitter", 0.05)
            p_h = 0.5 if aug.get("hflip", True) else 0.0
            p_v = 0.2 if aug.get("vflip", False) else 0.0
        elif strength == "medium":
            rot = max(15, aug.get("rotation_deg", 10))
            jitter = max(0.10, aug.get("color_jitter", 0.05))
            p_h = 0.5 if aug.get("hflip", True) else 0.0
            p_v = 0.3 if aug.get("vflip", False) else 0.0
        else:  # "chaos"
            rot = max(30, aug.get("rotation_deg", 10))
            jitter = max(0.25, aug.get("color_jitter", 0.05))
            p_h = 0.5
            p_v = 0.5

        if p_h > 0:
            base.append(T.RandomHorizontalFlip(p=p_h))
        if p_v > 0:
            base.append(T.RandomVerticalFlip(p=p_v))
        if rot > 0:
            base.append(T.RandomRotation(degrees=rot))
        if jitter and jitter > 0:
            base.append(T.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter, hue=min(0.1, jitter)
            ))

    base += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(base)
