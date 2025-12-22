import torch


def build_optimizer(model: torch.nn.Module, cfg):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )


__all__ = ["build_optimizer"]
