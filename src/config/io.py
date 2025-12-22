import yaml
from typing import Any, Dict, Optional

from .defaults import DEFAULT_CONFIG


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict `base` with dict `override`."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config and merge with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if path is None:
        return cfg
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(cfg, user_cfg)
    return cfg
