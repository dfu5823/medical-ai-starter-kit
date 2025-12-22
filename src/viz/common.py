import torch
import matplotlib.pyplot as plt


def _unnormalize_img(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = x.detach().cpu()
    return torch.clamp(x * std + mean, 0.0, 1.0)


def save_fig(fig: plt.Figure, out_path: str) -> None:
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = ["_unnormalize_img", "save_fig"]
