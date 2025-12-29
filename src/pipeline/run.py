"""Experiment orchestrator: load config, build data/model, train, evaluate, and log outputs."""

import os
from typing import Optional

import torch

from src.config import load_config
from src.utils import set_seed, ensure_dir, now_str, get_device
from src.logging import init_wandb, wandb_log_safe
from src.dataset import load_dermamnist, subset_dataset, build_loaders
from src.models import build_model, get_trainable_params
from src.training import train_one_epoch, validate_one_epoch, build_optimizer
from src.evaluation import evaluate, compute_metrics_multiclass
from src.viz import (
    save_fig,
    fig_sample_images_by_class,
    fig_train_curves,
    fig_confusion_matrix,
    fig_roc_ovr,
    fig_most_confident_wrong,
    cache_validation_examples,
)

# Optional W&B import for Image logging
try:
    import wandb
except Exception:
    wandb = None


def run_experiment(config_path: Optional[str] = None):
    """Execute a full training + eval cycle based on a YAML config.

    Steps:
    1) Load config, seed everything, and prepare the output directory.
    2) Build dataloaders (with optional dataset subsetting).
    3) Build the model, loss, and optimizer.
    4) Optionally log a gallery of sample images.
    5) Train for N epochs with live logging.
    6) Evaluate on validation data, compute metrics, and log artifacts/figures.
    7) Close W&B run (if enabled) and return metrics.
    """
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))

    out_dir = cfg["outputs"]["out_dir"]
    ensure_dir(out_dir)

    device = get_device(cfg.get("device", "cpu"))
    print(f"[INFO] Using device: {device}")

    run = init_wandb(cfg)

    # Data
    train_ds, val_ds, class_names = load_dermamnist(cfg)
    train_ds = subset_dataset(train_ds, int(cfg["data"]["subset_train"]), seed=int(cfg["seed"]))
    val_ds = subset_dataset(val_ds, int(cfg["data"]["subset_val"]), seed=int(cfg["seed"]) + 1)
    train_loader, val_loader = build_loaders(cfg, train_ds, val_ds)

    num_classes = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Model + optimizer
    model = build_model(cfg, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    # Optional sample images logging
    if cfg["outputs"].get("log_sample_images", True):
        try:
            fig = fig_sample_images_by_class(
                dataset=train_ds,
                class_names=class_names,
                n_per_class=int(cfg["outputs"]["n_samples_per_class"]),
            )
            if cfg["outputs"].get("save_figures", True):
                save_fig(fig, os.path.join(out_dir, f"samples_by_class_{now_str()}.png"))
            if run is not None and wandb is not None:
                wandb_log_safe(run, {"viz/samples_by_class": wandb.Image(fig)}, step=0)
        except Exception as e:
            print(f"[WARN] Could not log sample images: {e}")

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"]["log_every_steps"])
    print("Training started...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_every_steps=log_every,
            run=run,
            epoch=epoch,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        wandb_log_safe(
            run,
            {
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "epoch": epoch,
            },
            step=epoch,
        )

    # Full evaluation
    print("Evaluating final model on validation set...")
    y_true, y_pred, y_prob = evaluate(model, val_loader, device=device)
    metrics = compute_metrics_multiclass(y_true, y_pred, y_prob)
    print("[INFO] Final metrics:")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  macro_f1:  {metrics['macro_f1']:.4f}")
    print(f"  macro_auc: {metrics['macro_auc_ovr']:.4f}")

    wandb_log_safe(
        run,
        {
            "metrics/accuracy": metrics["accuracy"],
            "metrics/macro_f1": metrics["macro_f1"],
            "metrics/macro_auc_ovr": metrics["macro_auc_ovr"],
        },
        step=epochs,
    )
    for c, a in metrics["per_class_auc_ovr"].items():
        name = class_names[int(c)]
        wandb_log_safe(run, {f"metrics/auc_ovr/{name}": a}, step=epochs)

    # Figures
    if cfg["outputs"].get("log_confusion_matrix", True):
        fig_cm = fig_confusion_matrix(metrics["confusion_matrix"], class_names)
        if cfg["outputs"].get("save_figures", True):
            save_fig(fig_cm, os.path.join(out_dir, f"confusion_matrix_{now_str()}.png"))
        if run is not None and wandb is not None:
            wandb_log_safe(run, {"fig/confusion_matrix": wandb.Image(fig_cm)}, step=epochs)

    if cfg["outputs"].get("log_roc_plot", True):
        fig_roc = fig_roc_ovr(y_true, y_prob, class_names)
        if cfg["outputs"].get("save_figures", True):
            save_fig(fig_roc, os.path.join(out_dir, f"roc_ovr_{now_str()}.png"))
        if run is not None and wandb is not None:
            wandb_log_safe(run, {"fig/roc_ovr": wandb.Image(fig_roc)}, step=epochs)

    if cfg["outputs"].get("log_train_curves", True):
        fig_curves = fig_train_curves(history)
        if cfg["outputs"].get("save_figures", True):
            save_fig(fig_curves, os.path.join(out_dir, f"train_curves_{now_str()}.png"))
        if run is not None and wandb is not None:
            wandb_log_safe(run, {"fig/train_curves": wandb.Image(fig_curves)}, step=epochs)

    if cfg["outputs"].get("log_mistake_gallery", True):
        try:
            x_cache, y_cache, pred_cache, prob_cache = cache_validation_examples(
                model, val_loader, device=device, max_images=512
            )
            fig_wrong = fig_most_confident_wrong(
                x_batch=x_cache,
                y_true=y_cache,
                y_pred=pred_cache,
                y_prob=prob_cache,
                class_names=class_names,
                n=int(cfg["outputs"]["n_mistakes"]),
            )
            if cfg["outputs"].get("save_figures", True):
                save_fig(fig_wrong, os.path.join(out_dir, f"most_confident_wrong_{now_str()}.png"))
            if run is not None and wandb is not None:
                wandb_log_safe(run, {"viz/most_confident_wrong": wandb.Image(fig_wrong)}, step=epochs)
        except Exception as e:
            print(f"[WARN] Could not create mistake gallery: {e}")

    if run is not None:
        run.finish()

    print(f"[DONE] Outputs saved to: {out_dir}")
    return metrics


__all__ = ["run_experiment"]
