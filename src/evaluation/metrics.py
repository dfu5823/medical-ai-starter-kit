from typing import Any, Dict
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, accuracy_score


def compute_metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Minimal multiclass metrics:
    - accuracy
    - macro F1
    - per-class OvR AUC + macro AUC
    """
    num_classes = y_prob.shape[1]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    per_class_auc = {}
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class_auc[c] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, c])
        per_class_auc[c] = float(auc(fpr, tpr))

    macro_auc = float(np.nanmean(list(per_class_auc.values()))) if len(per_class_auc) else float("nan")
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_auc_ovr": float(macro_auc),
        "per_class_auc_ovr": per_class_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
    }


__all__ = ["compute_metrics_multiclass"]
