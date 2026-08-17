"""Guardrail: adding GPU/deep models must not shift the frozen sklearn baselines.

Phase A-D results were produced with scikit-learn defaults. If a later change
starts passing an argument that sklearn would otherwise choose itself (for
example ``early_stopping``), previously frozen numbers stop reproducing.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.sklearn_models import fit_model, resolve_use_gpu


def _toy_data(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 3, size=n)
    return X, y


def test_hist_gradient_boosting_keeps_sklearn_early_stopping_default():
    X, y = _toy_data()
    model = fit_model("hist_gradient_boosting", X, y, {"hist_gradient_boosting": {}})

    assert model.early_stopping == HistGradientBoostingClassifier().early_stopping


def test_hist_gradient_boosting_honours_an_explicit_early_stopping_setting():
    X, y = _toy_data()
    model = fit_model("hist_gradient_boosting", X, y, {"hist_gradient_boosting": {"early_stopping": True}})

    assert model.early_stopping is True


def test_logistic_regression_defaults_are_unchanged():
    X, y = _toy_data()
    model = fit_model("logistic_regression", X, y, {})

    assert model.C == 1.0
    assert model.max_iter == 2000
    assert model.class_weight == "balanced"


def test_random_forest_defaults_are_unchanged():
    X, y = _toy_data()
    model = fit_model("random_forest", X, y, {})

    assert model.n_estimators == 300
    assert model.class_weight == "balanced"
    assert model.random_state == 42


def test_per_model_gpu_flag_overrides_the_global_default():
    config = {"use_gpu": True, "lightgbm": {"use_gpu": False}, "xgboost": {}}

    assert resolve_use_gpu(config, "xgboost") is True
    assert resolve_use_gpu(config, "lightgbm") is False
    assert resolve_use_gpu({}, "xgboost") is False


def test_sample_weight_reaches_models_that_support_it():
    X, y = _toy_data()
    weights = np.linspace(0.1, 2.0, len(y))

    unweighted = fit_model("logistic_regression", X, y, {})
    weighted = fit_model("logistic_regression", X, y, {}, sample_weight=weights)

    assert not np.allclose(unweighted.coef_, weighted.coef_)
