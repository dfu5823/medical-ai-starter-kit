import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from .common import _unnormalize_img


def fig_sample_images_by_class(
    dataset,
    class_names: List[str],
    n_per_class: int,
    max_classes: Optional[int] = None,
) -> plt.Figure:
    """
    Log a small grid of sample images by class (from the *transformed* dataset).
    Note: these are post-transform (i.e., resized, normalized) visualized un-normalized.
    """
    num_classes = len(class_names)
    if max_classes is not None:
        num_classes = min(num_classes, max_classes)

    per_class_idxs = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):
        _, y = dataset[i]
        c = int(y.squeeze().item()) if torch.is_tensor(y) else int(np.array(y).squeeze())
        if c in per_class_idxs and len(per_class_idxs[c]) < n_per_class:
            per_class_idxs[c].append(i)
        if all(len(per_class_idxs[c]) >= n_per_class for c in per_class_idxs):
            break

    rows = num_classes
    cols = n_per_class
    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))

    plot_idx = 1
    for c in range(num_classes):
        for j in range(n_per_class):
            ax = fig.add_subplot(rows, cols, plot_idx)
            plot_idx += 1
            if j >= len(per_class_idxs[c]):
                ax.axis("off")
                continue
            x, _ = dataset[per_class_idxs[c][j]]
            x_vis = _unnormalize_img(x)
            ax.imshow(np.transpose(x_vis.numpy(), (1, 2, 0)))
            ax.set_title(class_names[c], fontsize=9)
            ax.axis("off")

    fig.suptitle("Sample Images by Class (post-transform)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def fig_most_confident_wrong(
    x_batch: torch.Tensor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    n: int
) -> plt.Figure:
    """
    Create a gallery of the most confident wrong predictions from a cached validation batch set.
    Inputs:
      x_batch: (N, 3, H, W) tensor corresponding to y_true/y_pred/y_prob
      y_true, y_pred: (N,)
      y_prob: (N, C)
    """
    wrong = (y_true != y_pred)
    wrong_idxs = np.where(wrong)[0]
    if len(wrong_idxs) == 0:
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No wrong predictions found in the cached set.",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        return fig

    conf = y_prob[np.arange(len(y_prob)), y_pred]
    wrong_conf = conf[wrong_idxs]
    top = wrong_idxs[np.argsort(-wrong_conf)][:n]

    cols = 6
    rows = int(math.ceil(len(top) / cols))
    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))

    for i, idx in enumerate(top):
        ax = fig.add_subplot(rows, cols, i + 1)
        x = _unnormalize_img(x_batch[idx])
        ax.imshow(np.transpose(x.numpy(), (1, 2, 0)))
        t = class_names[int(y_true[idx])]
        p = class_names[int(y_pred[idx])]
        c = conf[idx]
        ax.set_title(f"T:{t}\nP:{p} ({c:.2f})", fontsize=8)
        ax.axis("off")

    fig.suptitle("Most Confident Wrong Predictions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


@torch.no_grad()
def cache_validation_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_images: int = 512
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cache up to max_images validation examples (x, y_true, y_pred, y_prob) for galleries.
    Keeps memory reasonable for CPU.
    """
    model.eval()
    xs, ys, preds, probs = [], [], [], []
    total = 0
    for x, y in loader:
        if total >= max_images:
            break
        x = x.to(device)
        y0 = y.squeeze().long().cpu().numpy()
        logits = model(x).cpu()
        p = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()

        x_cpu = x.detach().cpu()
        xs.append(x_cpu)
        ys.append(y0)
        preds.append(pred)
        probs.append(p)

        total += x_cpu.shape[0]

    x_all = torch.cat(xs, dim=0)[:max_images]
    y_all = np.concatenate(ys, axis=0)[:max_images]
    pred_all = np.concatenate(preds, axis=0)[:max_images]
    prob_all = np.concatenate(probs, axis=0)[:max_images]
    return x_all, y_all, pred_all, prob_all


__all__ = [
    "fig_sample_images_by_class",
    "fig_most_confident_wrong",
    "cache_validation_examples",
]
