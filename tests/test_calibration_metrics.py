import numpy as np

from src.evaluation import compute_metrics


def test_multiclass_calibration_metrics_are_reported():
    truth = np.array([0, 1, 2, 0, 1, 2])
    probability = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    metrics = compute_metrics(truth, probability.argmax(axis=1), probability, labels=[0, 1, 2])
    assert metrics["log_loss"] >= 0
    assert metrics["brier_score"] >= 0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

