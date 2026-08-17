"""Optuna hyperparameter search with subject-level validation.

Scientific guardrail
--------------------
Every function here selects hyperparameters using **training and validation
data only**. The locked temporal test set must never be passed to this module.
Two selection modes are supported:

``holdout``
    Score each trial on an explicit validation split.
``groupkfold``
    Score each trial with subject-grouped K-fold CV over the combined
    train+validation rows, so no subject spans a fold boundary.

The returned parameters are meant to be written into a config and used for a
single final fit, keeping the test evaluation a one-shot measurement.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.models.sklearn_models import fit_model

LOGGER = logging.getLogger(__name__)

# Metrics that should be maximized; everything else is minimized.
_MAXIMIZE = {"macro_f1", "roc_auc_ovr", "balanced_accuracy", "accuracy"}

# Models with a defined search space. Validated before the study starts so a
# config typo fails immediately instead of surfacing as "no completed trials".
SEARCH_SPACES = frozenset(
    {
        "xgboost",
        "lightgbm",
        "hist_gradient_boosting",
        "random_forest",
        "logistic_regression",
        "torch_mlp",
        "ft_transformer",
    }
)


def _suggest(trial, model_name: str, use_gpu: bool, seed: int) -> dict[str, Any]:
    """Return a sampled hyperparameter dict for one Optuna trial."""
    if model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }
    elif model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }
    elif model_name == "hist_gradient_boosting":
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        }
    elif model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        }
    elif model_name == "logistic_regression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "max_iter": 4000,
            "class_weight": "balanced",
        }
    elif model_name == "torch_mlp":
        n_layers = trial.suggest_int("n_layers", 1, 4)
        width = trial.suggest_categorical("width", [64, 128, 256, 512])
        params = {
            "hidden_sizes": tuple(max(16, width // (2**index)) for index in range(n_layers)),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "max_epochs": 200,
            "patience": 20,
        }
    elif model_name == "ft_transformer":
        params = {
            "d_token": trial.suggest_categorical("d_token", [32, 64, 128]),
            "n_blocks": trial.suggest_int("n_blocks", 1, 4),
            "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
            "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.4),
            "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "max_epochs": 150,
            "patience": 16,
        }
    else:
        raise ValueError(f"No search space defined for model: {model_name}")

    params["random_state"] = seed
    params["use_gpu"] = use_gpu
    if model_name in {"xgboost", "lightgbm", "torch_mlp", "ft_transformer"}:
        params["early_stopping"] = True
    return params


def _score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str,
    labels: list[int],
) -> float:
    """Compute one selection metric from predicted probabilities."""
    y_pred = y_proba.argmax(axis=1)
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "log_loss":
        return float(log_loss(y_true, y_proba, labels=labels))
    if metric == "roc_auc_ovr":
        if len(labels) == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels))
    raise ValueError(f"Unsupported tuning metric: {metric}")


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    n_trials: int = 50,
    timeout: float | None = None,
    metric: str = "macro_f1",
    mode: str = "holdout",
    n_splits: int = 5,
    use_gpu: bool = False,
    seed: int = 42,
    study_name: str | None = None,
    storage: str | None = None,
    log_mlflow: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Search hyperparameters for one model and return the best configuration.

    Parameters
    ----------
    mode
        ``"holdout"`` scores trials on ``X_val``/``y_val``.
        ``"groupkfold"`` scores trials with subject-grouped CV over
        ``X_train``, using ``groups`` as the subject identifier.

    Returns
    -------
    dict
        ``{"model", "best_params", "best_value", "metric", "n_trials",
        "direction", "trials"}``. ``best_params`` is ready to paste into the
        ``models.<name>`` block of a config.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for tuning. pip install optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"No search space defined for model: {model_name}. Available: {', '.join(sorted(SEARCH_SPACES))}"
        )
    if mode == "holdout" and (X_val is None or y_val is None):
        raise ValueError("mode='holdout' requires X_val and y_val.")
    if mode == "groupkfold" and groups is None:
        raise ValueError("mode='groupkfold' requires subject groups.")
    if mode not in {"holdout", "groupkfold"}:
        raise ValueError(f"Unsupported tuning mode: {mode}")

    labels = sorted(np.unique(y_train).tolist())
    direction = "maximize" if metric in _MAXIMIZE else "minimize"

    def objective(trial) -> float:
        params = _suggest(trial, model_name, use_gpu, seed)
        models_config = {model_name: params}

        if mode == "holdout":
            estimator = fit_model(
                model_name,
                X_train,
                y_train,
                models_config,
                sample_weight=sample_weight,
                X_val=X_val,
                y_val=y_val,
            )
            proba = estimator.predict_proba(X_val)
            return _score(np.asarray(y_val), proba, metric, labels)

        splitter = GroupKFold(n_splits=n_splits)
        scores: list[float] = []
        for fold, (fit_index, score_index) in enumerate(splitter.split(X_train, y_train, groups=groups)):
            fold_weight = None if sample_weight is None else sample_weight[fit_index]
            estimator = fit_model(
                model_name,
                X_train[fit_index],
                y_train[fit_index],
                models_config,
                sample_weight=fold_weight,
                X_val=X_train[score_index],
                y_val=y_train[score_index],
            )
            proba = estimator.predict_proba(X_train[score_index])
            scores.append(_score(y_train[score_index], proba, metric, labels))

            # Report intermediate fold scores so the pruner can stop early.
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        if mode == "groupkfold"
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name or f"{model_name}_{metric}",
        storage=storage,
        load_if_exists=storage is not None,
    )

    LOGGER.info(
        "Tuning %s: %d trials, metric=%s (%s), mode=%s, gpu=%s",
        model_name,
        n_trials,
        metric,
        direction,
        mode,
        use_gpu,
    )
    # Individual trials may fail on an unlucky parameter combination (e.g. a
    # CUDA OOM at a large batch size); those are tolerated. A study where
    # *every* trial fails is a real error, so surface it rather than letting
    # Optuna report the confusing "no completed trials".
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=progress,
        catch=(RuntimeError,),
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} {model_name} trials failed. "
            "Re-run with logging at INFO to see the per-trial errors."
        )

    best_params = _suggest(_FixedTrial(study.best_params), model_name, use_gpu, seed)
    result = {
        "model": model_name,
        "metric": metric,
        "direction": direction,
        "mode": mode,
        "best_value": float(study.best_value),
        "best_params": _jsonable(best_params),
        "n_trials": len(study.trials),
        "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
    }

    if log_mlflow:
        from src.experiment import end_run, log_metrics, log_params, start_run

        start_run(run_name=f"tune_{model_name}", tags={"task": "hyperparameter_search"})
        log_params({f"best_{k}": v for k, v in result["best_params"].items()})
        log_metrics({f"best_{metric}": result["best_value"], "n_trials": result["n_trials"]})
        end_run()

    LOGGER.info("Best %s = %.4f after %d trials.", metric, study.best_value, len(study.trials))
    return result


class _FixedTrial:
    """Replay Optuna's best parameters through the same ``_suggest`` code path.

    Using one code path for sampling and replay guarantees the returned config
    matches what was actually evaluated, including any derived values such as
    ``hidden_sizes``.
    """

    def __init__(self, params: dict[str, Any]):
        self._params = params

    def suggest_int(self, name, low, high, step=1, log=False):
        return self._params[name]

    def suggest_float(self, name, low, high, step=None, log=False):
        return self._params[name]

    def suggest_categorical(self, name, choices):
        return self._params[name]


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy scalars and tuples so the result is YAML/JSON friendly."""
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            clean[key] = list(value)
        elif isinstance(value, np.integer):
            clean[key] = int(value)
        elif isinstance(value, np.floating):
            clean[key] = float(value)
        else:
            clean[key] = value
    return clean


def write_tuning_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    """Persist a tuning result as JSON and return the written path."""
    directory = Path(output_dir) / "tuning"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"best_params_{result['model']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def summarize_studies(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison table across per-model tuning results."""
    return pd.DataFrame(
        [
            {
                "model": r["model"],
                "metric": r["metric"],
                "best_value": r["best_value"],
                "n_trials": r["n_trials"],
                "n_pruned": r.get("n_pruned", 0),
            }
            for r in results
        ]
    ).sort_values("best_value", ascending=False)


def tune_many(
    model_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_dir: str | Path,
    *,
    tune_fn: Callable[..., dict[str, Any]] = tune_model,
    **kwargs: Any,
) -> pd.DataFrame:
    """Tune several models in sequence and write a comparison table."""
    results = []
    for name in model_names:
        try:
            result = tune_fn(name, X_train, y_train, **kwargs)
        except ImportError as exc:
            LOGGER.warning("Skipping %s: %s", name, exc)
            continue
        write_tuning_result(result, output_dir)
        results.append(result)

    summary = summarize_studies(results)
    directory = Path(output_dir) / "tuning"
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "tuning_summary.csv", index=False)
    return summary
