from typing import Any, Dict, Optional

# Optional W&B (code gracefully falls back if not installed or disabled)
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


def init_wandb(cfg: Dict[str, Any]) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initialize W&B if enabled and available. Returns run or None."""
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", True):
        return None
    if not _WANDB_AVAILABLE:
        print("[WARN] wandb not installed; continuing without W&B.")
        return None
    mode = wb.get("mode", "online")
    if mode == "disabled":
        return None

    wandb_kwargs = {
        "project": wb.get("project", "mdai-demo"),
        "name": wb.get("run_name", None),
        "tags": wb.get("tags", None),
        "config": cfg,
    }
    if wb.get("entity"):
        wandb_kwargs["entity"] = wb["entity"]
    if mode == "offline":
        import os
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(**wandb_kwargs)
    return run


def wandb_log_safe(run, data: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is None:
        return
    if "step" in data:
        data = dict(data)
        data.pop("step")
    run.log(data, step=step)


__all__ = ["init_wandb", "wandb_log_safe", "_WANDB_AVAILABLE"]
