from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

LOGGER = logging.getLogger(__name__)

# Models that accept sample_weight in fit(). Phase D relies on this for
# subject-balanced pair weighting, so anything used there must be listed.
_SUPPORTS_SAMPLE_WEIGHT = {
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "svm_rbf",
    "xgboost",
    "lightgbm",
    "torch_mlp",
    "ft_transformer",
    "soft_voting",
}

# Models that can early-stop against an explicit validation set.
_SUPPORTS_EVAL_SET = {"xgboost", "lightgbm", "torch_mlp", "ft_transformer"}


def resolve_use_gpu(models_config: dict[str, Any], model_name: str) -> bool:
    """Resolve the GPU flag for one model.

    A per-model ``use_gpu`` key wins over the top-level ``models.use_gpu``
    default, so a config can enable GPU globally and opt one model out.
    """
    if not models_config:
        return False
    model_config = models_config.get(model_name) or {}
    if "use_gpu" in model_config:
        return bool(model_config["use_gpu"])
    return bool(models_config.get("use_gpu", False))


def fit_model(
    model_name: str,
    X_train,
    y_train,
    config: dict[str, Any],
    sample_weight=None,
    X_val=None,
    y_val=None,
) -> Any:
    """Fit a supported model and return the fitted estimator.

    ``config`` is the ``models`` block of a project config. GPU use is read
    from ``models.<name>.use_gpu`` falling back to ``models.use_gpu``.

    When *X_val* and *y_val* are both given and the model config sets
    ``early_stopping: true``, gradient-boosting and neural models stop on the
    supplied validation set. The validation set is only ever used to decide
    when to stop -- it is never merged into the training data.
    """
    model_config = (config.get(model_name) or {}) if config else {}
    gpu = resolve_use_gpu(config, model_name)
    early_stopping = bool(model_config.get("early_stopping", False))
    has_eval_set = X_val is not None and y_val is not None

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
        # Only pass early-stopping arguments when the config asks for them so
        # that sklearn's defaults (and therefore frozen results) are preserved.
        hgb_kwargs: dict[str, Any] = {
            "max_iter": model_config.get("max_iter", 200),
            "learning_rate": model_config.get("learning_rate", 0.05),
            "max_leaf_nodes": model_config.get("max_leaf_nodes", 31),
            "random_state": model_config.get("random_state", 42),
        }
        if "early_stopping" in model_config:
            hgb_kwargs["early_stopping"] = model_config["early_stopping"]
        if "validation_fraction" in model_config:
            hgb_kwargs["validation_fraction"] = model_config["validation_fraction"]
        if "l2_regularization" in model_config:
            hgb_kwargs["l2_regularization"] = model_config["l2_regularization"]
        if "min_samples_leaf" in model_config:
            hgb_kwargs["min_samples_leaf"] = model_config["min_samples_leaf"]
        model = HistGradientBoostingClassifier(**hgb_kwargs)

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

        xgb_kwargs: dict[str, Any] = {
            "n_estimators": model_config.get("n_estimators", 200),
            "learning_rate": model_config.get("learning_rate", 0.05),
            "max_depth": model_config.get("max_depth", 3),
            "subsample": model_config.get("subsample", 1.0),
            "colsample_bytree": model_config.get("colsample_bytree", 1.0),
            "min_child_weight": model_config.get("min_child_weight", 1),
            "reg_lambda": model_config.get("reg_lambda", 1.0),
            "reg_alpha": model_config.get("reg_alpha", 0.0),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": model_config.get("random_state", 42),
            "n_jobs": model_config.get("n_jobs", -1),
            "verbosity": model_config.get("verbosity", 0),
        }
        if gpu:
            xgb_kwargs["device"] = "cuda"
            xgb_kwargs["tree_method"] = "hist"
            LOGGER.info("xgboost: device=cuda tree_method=hist")
        if early_stopping and has_eval_set:
            xgb_kwargs["early_stopping_rounds"] = model_config.get("early_stopping_rounds", 20)
        model = XGBClassifier(**xgb_kwargs)

    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is optional and not installed.") from exc

        lgbm_kwargs: dict[str, Any] = {
            "n_estimators": model_config.get("n_estimators", 200),
            "learning_rate": model_config.get("learning_rate", 0.05),
            "num_leaves": model_config.get("num_leaves", 31),
            "max_depth": model_config.get("max_depth", -1),
            "subsample": model_config.get("subsample", 1.0),
            "colsample_bytree": model_config.get("colsample_bytree", 1.0),
            "min_child_samples": model_config.get("min_child_samples", 20),
            "reg_lambda": model_config.get("reg_lambda", 0.0),
            "reg_alpha": model_config.get("reg_alpha", 0.0),
            "random_state": model_config.get("random_state", 42),
            "n_jobs": model_config.get("n_jobs", -1),
            "verbose": model_config.get("verbosity", -1),
        }
        if gpu:
            lgbm_kwargs["device"] = "gpu"
            LOGGER.info("lightgbm: device=gpu")
        model = LGBMClassifier(**lgbm_kwargs)

    elif model_name in {"torch_mlp", "ft_transformer"}:
        import inspect

        from src.models.deep_tabular import FTTransformerClassifier, TabularMLPClassifier

        builder = TabularMLPClassifier if model_name == "torch_mlp" else FTTransformerClassifier
        valid = set(inspect.signature(builder.__init__).parameters) - {"self"}
        ignored = set(model_config) - valid - {"use_gpu", "early_stopping"}
        if ignored:
            LOGGER.warning("%s: ignoring unsupported config keys %s", model_name, sorted(ignored))
        kwargs = {key: value for key, value in model_config.items() if key in valid}
        kwargs["device"] = "cuda" if gpu else "cpu"
        kwargs["early_stopping"] = early_stopping
        if "hidden_sizes" in kwargs:
            kwargs["hidden_sizes"] = tuple(kwargs["hidden_sizes"])
        model = builder(**kwargs)

    elif model_name == "soft_voting":
        from src.models.ensemble import build_soft_voting

        members = model_config.get("members")
        if not members:
            raise ValueError("soft_voting requires a non-empty 'members' list in config.")
        # build_soft_voting fits each member itself, so return directly.
        return build_soft_voting(
            members,
            X_train,
            y_train,
            config,
            sample_weight=sample_weight,
            X_val=X_val,
            y_val=y_val,
            weights=model_config.get("weights"),
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # -- fit ---------------------------------------------------------------
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None and model_name in _SUPPORTS_SAMPLE_WEIGHT:
        fit_kwargs["sample_weight"] = sample_weight
    elif sample_weight is not None:
        LOGGER.warning("%s does not support sample_weight; ignoring it.", model_name)

    if early_stopping and has_eval_set and model_name in _SUPPORTS_EVAL_SET:
        if model_name == "xgboost":
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        elif model_name == "lightgbm":
            import lightgbm as lgb

            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(model_config.get("early_stopping_rounds", 20), verbose=False)]
        else:  # torch_mlp / ft_transformer
            fit_kwargs["eval_set"] = (X_val, y_val)

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def predict_model(model: Any, X_test) -> np.ndarray:
    """Predict labels."""
    return model.predict(X_test)


def predict_proba_model(model: Any, X_test) -> np.ndarray | None:
    """Predict probabilities when the model supports them."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    return None
