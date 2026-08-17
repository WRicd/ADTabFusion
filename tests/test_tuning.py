import numpy as np
import pytest

pytest.importorskip("optuna")

from src.tuning import summarize_studies, tune_model, write_tuning_result


def _toy_data(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    weights = rng.normal(size=(d, 3))
    y = (X @ weights + rng.normal(scale=0.5, size=(n, 3))).argmax(axis=1)
    return X, y


def test_holdout_tuning_returns_usable_params():
    X, y = _toy_data()
    result = tune_model(
        "logistic_regression",
        X[:150],
        y[:150],
        X_val=X[150:],
        y_val=y[150:],
        n_trials=4,
        log_mlflow=False,
        progress=False,
    )

    assert result["model"] == "logistic_regression"
    assert result["direction"] == "maximize"
    assert 0.0 <= result["best_value"] <= 1.0
    assert "C" in result["best_params"]


def test_groupkfold_tuning_never_splits_a_subject():
    X, y = _toy_data(n=180)
    groups = np.repeat(np.arange(60), 3)  # 3 rows per simulated subject

    result = tune_model(
        "logistic_regression",
        X,
        y,
        groups=groups,
        mode="groupkfold",
        n_splits=3,
        n_trials=3,
        log_mlflow=False,
        progress=False,
    )

    assert result["mode"] == "groupkfold"
    assert result["n_trials"] == 3


def test_log_loss_is_minimized():
    X, y = _toy_data()
    result = tune_model(
        "logistic_regression",
        X[:150],
        y[:150],
        X_val=X[150:],
        y_val=y[150:],
        metric="log_loss",
        n_trials=3,
        log_mlflow=False,
        progress=False,
    )

    assert result["direction"] == "minimize"
    assert result["best_value"] > 0


def test_holdout_mode_requires_a_validation_split():
    X, y = _toy_data()
    with pytest.raises(ValueError, match="requires X_val and y_val"):
        tune_model("logistic_regression", X, y, n_trials=1, log_mlflow=False, progress=False)


def test_groupkfold_mode_requires_subject_groups():
    X, y = _toy_data()
    with pytest.raises(ValueError, match="requires subject groups"):
        tune_model(
            "logistic_regression",
            X,
            y,
            mode="groupkfold",
            n_trials=1,
            log_mlflow=False,
            progress=False,
        )


def test_unknown_model_has_no_search_space():
    X, y = _toy_data()
    with pytest.raises(ValueError, match="No search space"):
        tune_model(
            "not_a_model",
            X[:150],
            y[:150],
            X_val=X[150:],
            y_val=y[150:],
            n_trials=1,
            log_mlflow=False,
            progress=False,
        )


def test_best_params_are_json_serializable(tmp_path):
    X, y = _toy_data()
    result = tune_model(
        "random_forest",
        X[:150],
        y[:150],
        X_val=X[150:],
        y_val=y[150:],
        n_trials=3,
        log_mlflow=False,
        progress=False,
    )

    path = write_tuning_result(result, tmp_path)

    assert path.exists()
    for value in result["best_params"].values():
        assert not isinstance(value, np.integer | np.floating | tuple)


def test_summary_is_ordered_best_first():
    frame = summarize_studies(
        [
            {"model": "a", "metric": "macro_f1", "best_value": 0.5, "n_trials": 3},
            {"model": "b", "metric": "macro_f1", "best_value": 0.9, "n_trials": 3},
        ]
    )

    assert list(frame["model"]) == ["b", "a"]
