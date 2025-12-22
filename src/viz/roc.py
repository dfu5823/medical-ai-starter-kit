import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def fig_roc_ovr(y_true: np.ndarray, y_prob: np.ndarray, class_names):
    """Plot one-vs-rest ROC curves for each class on one plot."""
    num_classes = y_prob.shape[1]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1], linestyle="--")

    aucs = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, c])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={a:.2f})")

    macro_auc = np.mean(aucs) if len(aucs) else float("nan")
    ax.set_title(f"OvR ROC Curves (macro AUC={macro_auc:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


__all__ = ["fig_roc_ovr"]
