from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def fit_model(
    model_name: str,
    X_train,
    y_train,
    config: dict[str, Any],
    sample_weight=None,
) -> Any:
    """Fit a supported sklearn model."""
    model_config = config.get(model_name, {}) if config else {}
    if model_name == "logistic_regression":
        model = LogisticRegression(
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 2000),
            class_weight=model_config.get("class_weight", "balanced"),
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=model_config.get("n_estimators", 300),
            max_depth=model_config.get("max_depth"),
            class_weight=model_config.get("class_weight", "balanced"),
            random_state=model_config.get("random_state", 42),
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=model_config.get("max_iter", 200),
            learning_rate=model_config.get("learning_rate", 0.05),
            max_leaf_nodes=model_config.get("max_leaf_nodes", 31),
            random_state=model_config.get("random_state", 42),
        )
    elif model_name == "svm_rbf":
        model = SVC(
            C=model_config.get("C", 1.0),
            gamma=model_config.get("gamma", "scale"),
            class_weight=model_config.get("class_weight", "balanced"),
            probability=True,
        )
    elif model_name == "mlp_sklearn":
        model = MLPClassifier(
            hidden_layer_sizes=tuple(model_config.get("hidden_layer_sizes", [64, 32])),
            max_iter=model_config.get("max_iter", 400),
            random_state=model_config.get("random_state", 42),
        )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is optional and not installed.") from exc
        model = XGBClassifier(
            n_estimators=model_config.get("n_estimators", 200),
            learning_rate=model_config.get("learning_rate", 0.05),
            max_depth=model_config.get("max_depth", 3),
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=model_config.get("random_state", 42),
        )
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is optional and not installed.") from exc
        model = LGBMClassifier(
            n_estimators=model_config.get("n_estimators", 200),
            learning_rate=model_config.get("learning_rate", 0.05),
            random_state=model_config.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def predict_model(model: Any, X_test) -> np.ndarray:
    """Predict labels."""
    return model.predict(X_test)


def predict_proba_model(model: Any, X_test) -> np.ndarray | None:
    """Predict probabilities when the model supports them."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    return None
