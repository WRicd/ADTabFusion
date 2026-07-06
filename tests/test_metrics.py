import numpy as np

from src.evaluation import compute_metrics


def test_metric_keys():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]])
    metrics = compute_metrics(y_true, y_pred, y_prob, labels=[0, 1])
    assert "accuracy" in metrics
    assert "macro_f1" in metrics

