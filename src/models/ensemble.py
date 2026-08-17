"""Soft-voting ensemble over already-fitted member estimators.

Unlike ``sklearn.ensemble.VotingClassifier``, this wrapper accepts members that
are **already fitted** on the preprocessed matrix. That matters here because
each member may need a different ``fit`` call -- GPU boosters take an eval set
for early stopping, the torch models take a validation tuple -- which the
project's :func:`src.models.sklearn_models.fit_model` dispatch already handles.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

LOGGER = logging.getLogger(__name__)


class SoftVotingClassifier(ClassifierMixin, BaseEstimator):
    """Average member probabilities, optionally weighted.

    Parameters
    ----------
    estimators
        ``(name, fitted_estimator)`` pairs. Every member must expose
        ``predict_proba`` and have been fitted on the same feature matrix.
    weights
        Optional per-member weights; defaults to uniform. Normalized
        internally so the averaged output stays on the probability simplex.
    """

    def __init__(
        self,
        estimators: Sequence[tuple[str, Any]],
        weights: Sequence[float] | None = None,
    ):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X=None, y=None, sample_weight=None):
        """Validate members; they are already fitted so no training happens."""
        if not self.estimators:
            raise ValueError("SoftVotingClassifier requires at least one member.")

        classes: np.ndarray | None = None
        for name, estimator in self.estimators:
            if not hasattr(estimator, "predict_proba"):
                raise ValueError(f"Ensemble member '{name}' has no predict_proba.")
            member_classes = getattr(estimator, "classes_", None)
            if member_classes is None:
                continue
            if classes is None:
                classes = np.asarray(member_classes)
            elif not np.array_equal(classes, np.asarray(member_classes)):
                raise ValueError(f"Ensemble member '{name}' has classes {member_classes!r}, expected {classes!r}.")

        if classes is None:
            if y is None:
                raise ValueError("Cannot infer classes_: pass y or fitted members.")
            classes = np.unique(y)

        self.classes_ = classes
        self.weights_ = self._normalized_weights()
        return self

    def _normalized_weights(self) -> np.ndarray:
        if self.weights is None:
            values = np.ones(len(self.estimators), dtype=float)
        else:
            if len(self.weights) != len(self.estimators):
                raise ValueError(f"Got {len(self.weights)} weights for {len(self.estimators)} members.")
            values = np.asarray(self.weights, dtype=float)
            if np.any(values < 0):
                raise ValueError("Ensemble weights must be non-negative.")
        total = values.sum()
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        return values / total

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError("Estimator is not fitted; call fit() first.")
        weights = self.weights_
        total: np.ndarray | None = None
        for weight, (_, estimator) in zip(weights, self.estimators):
            proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
            total = proba * weight if total is None else total + proba * weight
        assert total is not None
        # Guard against members whose rows drift slightly off the simplex.
        return total / total.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_soft_voting(
    member_names: Sequence[str],
    X_train,
    y_train,
    models_config: dict[str, Any],
    sample_weight=None,
    X_val=None,
    y_val=None,
    weights: Sequence[float] | None = None,
) -> SoftVotingClassifier:
    """Fit each member with the shared dispatch and wrap them in a vote.

    Members that raise :class:`ImportError` (an optional dependency is not
    installed) are skipped with a warning rather than failing the whole run.
    """
    from src.models.sklearn_models import fit_model

    fitted: list[tuple[str, Any]] = []
    kept_weights: list[float] = []
    for index, name in enumerate(member_names):
        try:
            estimator = fit_model(
                name,
                X_train,
                y_train,
                models_config,
                sample_weight=sample_weight,
                X_val=X_val,
                y_val=y_val,
            )
        except ImportError as exc:
            LOGGER.warning("Ensemble: skipping member '%s' (%s).", name, exc)
            continue
        fitted.append((name, estimator))
        kept_weights.append(1.0 if weights is None else float(weights[index]))

    if not fitted:
        raise RuntimeError("No ensemble members could be fitted.")

    LOGGER.info("Ensemble: %d members (%s).", len(fitted), ", ".join(n for n, _ in fitted))
    ensemble = SoftVotingClassifier(fitted, weights=None if weights is None else kept_weights)
    return ensemble.fit(y=y_train)
