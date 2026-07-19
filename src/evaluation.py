from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Compute core classification metrics."""
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            unique = np.unique(y_true)
            if len(unique) == 2:
                score = roc_auc_score(y_true, y_proba[:, 1])
            else:
                score = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro", labels=labels
                )
            result["roc_auc_ovr"] = float(score)
        except ValueError:
            result["roc_auc_ovr"] = None
        try:
            result["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
            one_hot = np.eye(y_proba.shape[1])[np.asarray(y_true, dtype=int)]
            result["brier_score"] = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))
        except (ValueError, IndexError):
            result["log_loss"] = None
            result["brier_score"] = None
    else:
        result["roc_auc_ovr"] = None
        result["log_loss"] = None
        result["brier_score"] = None

    metric_labels = labels or sorted(np.unique(y_true).tolist())
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=metric_labels, zero_division=0
    )
    for index, label in enumerate(metric_labels):
        result[f"precision_class_{label}"] = float(precision[index])
        result[f"recall_class_{label}"] = float(recall[index])
        result[f"f1_class_{label}"] = float(per_class_f1[index])
    result["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=metric_labels
    ).tolist()
    return result
