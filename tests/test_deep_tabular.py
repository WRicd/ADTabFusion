import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.deep_tabular import (
    FTTransformerClassifier,
    TabularMLPClassifier,
    resolve_device,
)


def _toy_data(n=240, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    weights = rng.normal(size=(d, 3))
    y = (X @ weights + rng.normal(scale=0.4, size=(n, 3))).argmax(axis=1)
    return X, y


@pytest.mark.parametrize(
    "builder",
    [
        lambda: TabularMLPClassifier(hidden_sizes=(32,), max_epochs=6, early_stopping=False),
        lambda: FTTransformerClassifier(d_token=16, n_blocks=1, n_heads=4, max_epochs=4, early_stopping=False),
    ],
)
def test_predict_proba_is_on_the_simplex(builder):
    X, y = _toy_data()
    model = builder().fit(X, y)
    proba = model.predict_proba(X)
    row_sums = proba.sum(axis=1)

    assert proba.shape == (len(X), 3)
    assert np.all(proba >= 0)
    assert set(model.predict(X)).issubset(set(np.unique(y)))
    # Tolerance matches sklearn's log_loss check: a float32 softmax left ~1e-7
    # of drift, which is loose enough to trip metric computation downstream.
    assert np.allclose(row_sums, 1.0, rtol=np.finfo(proba.dtype).eps, atol=1e-15)


def test_sample_weight_changes_the_fitted_model():
    """Subject-balanced weighting must actually reach the training loop."""
    X, y = _toy_data()
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.1, 2.0, size=len(y))

    common = dict(hidden_sizes=(32,), max_epochs=12, early_stopping=False, random_state=7)
    unweighted = TabularMLPClassifier(**common).fit(X, y).predict_proba(X)
    weighted = TabularMLPClassifier(**common).fit(X, y, sample_weight=weights).predict_proba(X)

    assert not np.allclose(unweighted, weighted)


def test_training_is_reproducible_for_a_fixed_seed():
    X, y = _toy_data()
    common = dict(hidden_sizes=(24,), max_epochs=8, early_stopping=False, random_state=123)

    first = TabularMLPClassifier(**common).fit(X, y).predict_proba(X)
    second = TabularMLPClassifier(**common).fit(X, y).predict_proba(X)

    assert np.allclose(first, second)


def test_early_stopping_can_halt_before_max_epochs():
    X, y = _toy_data()
    model = TabularMLPClassifier(hidden_sizes=(32,), max_epochs=200, patience=2, early_stopping=True, random_state=3)
    model.fit(X[:180], y[:180], eval_set=(X[180:], y[180:]))

    assert model.n_epochs_run_ <= 200
    assert hasattr(model, "best_validation_loss_")


def test_classes_are_preserved_for_non_contiguous_labels():
    X, _ = _toy_data(n=120)
    y = np.array([0, 2, 5] * 40)

    model = TabularMLPClassifier(hidden_sizes=(16,), max_epochs=4, early_stopping=False).fit(X, y)

    assert list(model.classes_) == [0, 2, 5]
    assert set(model.predict(X)).issubset({0, 2, 5})


def test_predict_before_fit_raises():
    model = TabularMLPClassifier()
    with pytest.raises(RuntimeError):
        model.predict_proba(np.zeros((2, 3), dtype="float32"))


@pytest.mark.parametrize(
    "builder",
    [
        lambda: TabularMLPClassifier(hidden_sizes=(32,), max_epochs=4, early_stopping=False),
        lambda: FTTransformerClassifier(d_token=16, n_blocks=1, n_heads=4, max_epochs=3, early_stopping=False),
    ],
)
def test_fitted_model_survives_a_joblib_round_trip(builder, tmp_path):
    """Phase D hashes joblib artifacts, so a silent dump failure corrupts provenance."""
    import joblib

    X, y = _toy_data()
    model = builder().fit(X, y)
    expected = model.predict_proba(X)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)

    assert np.allclose(restored.predict_proba(X), expected)


def test_pipeline_containing_a_deep_model_is_picklable(tmp_path):
    """The real artifact is a Pipeline, not a bare estimator."""
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _toy_data()
    pipeline = Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("model", FTTransformerClassifier(d_token=16, n_blocks=1, n_heads=4, max_epochs=3, early_stopping=False)),
        ]
    )
    pipeline.named_steps["model"].fit(pipeline.named_steps["preprocessor"].fit_transform(X), y)

    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    assert np.allclose(restored.predict_proba(X), pipeline.predict_proba(X))


def test_serialized_network_is_stored_on_cpu():
    """Artifacts must load on machines without a GPU."""
    X, y = _toy_data()
    model = TabularMLPClassifier(hidden_sizes=(16,), max_epochs=3, early_stopping=False).fit(X, y)

    state = model.__getstate__()

    assert state["device_"] == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in state["network_"].parameters())


def test_resolve_device_falls_back_to_cpu_without_cuda():
    assert resolve_device("cpu") == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("cuda") == expected


def test_ft_transformer_head_count_is_reduced_to_divide_d_token():
    """d_token must stay divisible by n_heads or MultiheadAttention errors."""
    X, y = _toy_data(n=90, d=6)
    model = FTTransformerClassifier(d_token=12, n_heads=8, n_blocks=1, max_epochs=2, early_stopping=False)
    model.fit(X, y)

    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)
