import torch


def get_device(device_str: str = "cpu") -> torch.device:
    """Resolve requested device string to a torch.device, defaulting to CPU."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
