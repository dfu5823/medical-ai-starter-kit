import matplotlib.pyplot as plt


def fig_train_curves(history):
    """Plot train/val loss and acc curves."""
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(history["train_loss"], label="train_loss")
    ax1.plot(history["val_loss"], label="val_loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="train_acc")
    ax2.plot(history["val_acc"], label="val_acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    return fig


__all__ = ["fig_train_curves"]
