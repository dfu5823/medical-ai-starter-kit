import os
import time


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
