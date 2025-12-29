import os
from typing import Any, Dict, List, Tuple

import medmnist
from medmnist import INFO

from .transforms import build_transforms
from ..utils import ensure_dir


def load_dermamnist(cfg: Dict[str, Any]):
    """
    Download/load DermMNIST.
    Returns train_dataset, val_dataset, class_names.
    """
    dataset_name = cfg["data"]["dataset"].lower()
    if dataset_name != "dermamnist":
        raise ValueError(f"This demo expects dermamnist, got: {dataset_name}")

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]

    root = os.path.abspath(os.path.expanduser(cfg["data"]["root"]))
    ensure_dir(root)  # medmnist expects the root directory to already exist
    download = bool(cfg["data"].get("download", True))

    train_tf = build_transforms(cfg, train=True)
    val_tf = build_transforms(cfg, train=False)

    train_ds = DataClass(split="train", root=root, transform=train_tf, download=download)
    val_ds = DataClass(split="val", root=root, transform=val_tf, download=download)

    return train_ds, val_ds, class_names


__all__ = ["load_dermamnist"]
