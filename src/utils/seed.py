import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Best-effort reproducibility on CPU (and CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
