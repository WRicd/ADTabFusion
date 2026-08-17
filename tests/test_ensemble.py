import numpy as np
import pytest

from src.models.ensemble import SoftVotingClassifier, build_soft_voting


class _StubMember:
    """Minimal fitted-estimator stand-in returning fixed probabilities."""

    def __init__(self, proba, classes=(0, 1, 2)):
        self._proba = np.asarray(proba, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return np.repeat(self._proba[None, :], len(X), axis=0)


def test_uniform_vote_averages_member_probabilities():
    ensemble = SoftVotingClassifier([("a", _StubMember([1.0, 0.0, 0.0])), ("b", _StubMember([0.0, 1.0, 0.0]))]).fit()

    proba = ensemble.predict_proba(np.zeros((3, 2)))

    assert np.allclose(proba, [[0.5, 0.5, 0.0]] * 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_weights_shift_the_decision():
    members = [("a", _StubMember([1.0, 0.0, 0.0])), ("b", _StubMember([0.0, 1.0, 0.0]))]

    balanced = SoftVotingClassifier(members).fit().predict(np.zeros((1, 2)))
    skewed = SoftVotingClassifier(members, weights=[9.0, 1.0]).fit().predict(np.zeros((1, 2)))

    assert skewed[0] == 0
    # Ties resolve to the first class, so only the skewed case is a real signal.
    assert balanced[0] in (0, 1)


def test_weights_are_normalized_so_output_stays_on_the_simplex():
    ensemble = SoftVotingClassifier(
        [("a", _StubMember([0.6, 0.3, 0.1])), ("b", _StubMember([0.2, 0.5, 0.3]))],
        weights=[7.0, 3.0],
    ).fit()

    proba = ensemble.predict_proba(np.zeros((4, 2)))

    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.allclose(proba[0], 0.7 * np.array([0.6, 0.3, 0.1]) + 0.3 * np.array([0.2, 0.5, 0.3]))


def test_mismatched_weight_count_is_rejected():
    ensemble = SoftVotingClassifier([("a", _StubMember([1.0, 0.0, 0.0]))], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="weights"):
        ensemble.fit()


def test_members_with_disagreeing_classes_are_rejected():
    ensemble = SoftVotingClassifier(
        [
            ("a", _StubMember([1.0, 0.0, 0.0], classes=(0, 1, 2))),
            ("b", _StubMember([1.0, 0.0, 0.0], classes=(0, 1, 9))),
        ]
    )
    with pytest.raises(ValueError, match="classes"):
        ensemble.fit()


def test_empty_ensemble_is_rejected():
    with pytest.raises(ValueError):
        SoftVotingClassifier([]).fit()


def test_predict_before_fit_raises():
    ensemble = SoftVotingClassifier([("a", _StubMember([1.0, 0.0, 0.0]))])
    with pytest.raises(RuntimeError):
        ensemble.predict_proba(np.zeros((1, 2)))


def test_build_soft_voting_fits_real_members():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = rng.integers(0, 3, size=80)

    ensemble = build_soft_voting(
        ["logistic_regression", "random_forest"],
        X,
        y,
        {"logistic_regression": {"max_iter": 200}, "random_forest": {"n_estimators": 10}},
    )
    proba = ensemble.predict_proba(X)

    assert [name for name, _ in ensemble.estimators] == ["logistic_regression", "random_forest"]
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_build_soft_voting_skips_members_whose_dependency_is_missing(monkeypatch):
    """A missing optional package should drop one member, not the whole run."""
    import src.models.sklearn_models as sklearn_models

    real_fit = sklearn_models.fit_model

    def fake_fit(model_name, *args, **kwargs):
        if model_name == "lightgbm":
            raise ImportError("lightgbm is optional and not installed.")
        return real_fit(model_name, *args, **kwargs)

    monkeypatch.setattr(sklearn_models, "fit_model", fake_fit)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = rng.integers(0, 2, size=60)

    ensemble = build_soft_voting(
        ["logistic_regression", "lightgbm"],
        X,
        y,
        {"logistic_regression": {"max_iter": 200}},
    )

    assert [name for name, _ in ensemble.estimators] == ["logistic_regression"]


def test_build_soft_voting_raises_when_no_member_survives(monkeypatch):
    import src.models.sklearn_models as sklearn_models

    def always_missing(model_name, *args, **kwargs):
        raise ImportError(f"{model_name} is not installed.")

    monkeypatch.setattr(sklearn_models, "fit_model", always_missing)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.integers(0, 2, size=20)

    with pytest.raises(RuntimeError, match="No ensemble members"):
        build_soft_voting(["xgboost", "lightgbm"], X, y, {})


def test_unknown_member_name_fails_loudly():
    """A config typo must not be silently skipped."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.integers(0, 2, size=20)

    with pytest.raises(ValueError, match="Unsupported model"):
        build_soft_voting(["logistic_regresion"], X, y, {})
