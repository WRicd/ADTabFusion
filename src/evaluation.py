from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
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
    else:
        result["roc_auc_ovr"] = None
    return result

