import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional, Tuple


def subset_dataset(ds, n: Optional[int], seed: int):
    """Optionally subset a dataset to first N examples with a shuffled index for variety."""
    if n is None or n <= 0 or n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    idx = idx[:n].tolist()
    return Subset(ds, idx)


def build_loaders(cfg, train_ds, val_ds) -> Tuple[DataLoader, DataLoader]:
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])
    pin = bool(cfg["data"]["pin_memory"])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    return train_loader, val_loader


__all__ = ["subset_dataset", "build_loaders"]
