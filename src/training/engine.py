from typing import Tuple

import torch

from src.logging import wandb_log_safe


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    log_every_steps: int,
    run=None,
    epoch: int = 0
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, avg_acc)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.squeeze().long().to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += correct
        total += bs

        if (step + 1) % log_every_steps == 0:
            wandb_log_safe(run, {
                "train/step_loss": loss.item(),
                "train/step_acc": correct / max(1, bs),
                "train/epoch": epoch,
            }, step=epoch * len(loader) + step)

    avg_loss = total_loss / max(1, total)
    avg_acc = total_correct / max(1, total)
    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validation over one epoch. Returns (avg_loss, avg_acc)."""
    model.eval()
    val_loss_sum = 0.0
    val_total = 0
    val_correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long().to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        val_correct += (preds == y).sum().item()
        bs = x.size(0)
        val_loss_sum += loss.item() * bs
        val_total += bs

    val_loss = val_loss_sum / max(1, val_total)
    val_acc = val_correct / max(1, val_total)
    return val_loss, val_acc


__all__ = ["train_one_epoch", "validate_one_epoch"]
