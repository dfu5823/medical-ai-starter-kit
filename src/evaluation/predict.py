from typing import Tuple
import numpy as np
import torch


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: (N,)
      y_pred: (N,)
      y_prob: (N, C) softmax probabilities
    """
    model.eval()
    ys, preds, probs = [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long()
        logits = model(x).cpu()
        p = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()

        ys.append(y.numpy())
        preds.append(pred)
        probs.append(p)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    return y_true, y_pred, y_prob


__all__ = ["evaluate"]
