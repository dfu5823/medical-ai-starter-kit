from typing import Any, Dict

# Default configuration for the DermMNIST workflow
DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cpu",  # keep CPU by default; user can set "cuda" if available
    "data": {
        "dataset": "dermamnist",
        "root": "./data",
        "download": True,
        "subset_train": 2000,
        "subset_val": 500,
        "num_workers": 0,
        "pin_memory": False,
        "image_size": 128,
    },
    "aug": {
        "aug_strength": "light",  # light | medium | chaos
        "hflip": True,
        "vflip": False,
        "rotation_deg": 10,
        "color_jitter": 0.05,
    },
    "model": {
        "backbone": "resnet18",
        "pretrained": True,
        "freeze_backbone": True,
        "dropout": 0.2,
    },
    "train": {
        "epochs": 2,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "log_every_steps": 50,
    },
    "outputs": {
        "out_dir": "./outputs",
        "save_figures": True,
        "log_train_curves": True,
        "log_confusion_matrix": True,
        "log_roc_plot": True,
        "log_sample_images": True,
        "log_mistake_gallery": True,
        "n_samples_per_class": 2,
        "n_mistakes": 12,
    },
    "wandb": {
        "enabled": True,
        "mode": "online",
        "project": "mdai-dermamnist-demo",
        "entity": None,
        "run_name": None,
        "tags": ["session1", "workflow-demo"],
    },
}
