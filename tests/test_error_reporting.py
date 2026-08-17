import logging

import numpy as np
import pytest

from src import experiment
from src.adni_inventory import _read_sample
from src.evaluation import compute_metrics


def test_experiment_context_marks_failed_run_and_reraises(monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "start_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "end_run", lambda status="FINISHED": calls.append(status))

    with pytest.raises(RuntimeError):
        with experiment.ExperimentContext("unit_test"):
            raise RuntimeError("training failed")

    assert calls == ["FAILED"]


def test_experiment_context_marks_finished_run(monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "start_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "end_run", lambda status="FINISHED": calls.append(status))

    with experiment.ExperimentContext("unit_test"):
        pass

    assert calls == ["FINISHED"]


def test_unreadable_csv_reports_every_attempted_encoding(tmp_path, caplog):
    path = tmp_path / "broken.csv"
    path.write_text("a,b\n1\n2,3,4\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(Exception):
            _read_sample(path)

    assert "utf-8-sig" in caplog.text and "latin-1" in caplog.text


def test_unavailable_probability_metrics_are_logged(caplog):
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0])
    y_proba = np.full((4, 2), 0.5)

    with caplog.at_level(logging.WARNING):
        result = compute_metrics(y_true, y_pred, y_proba, labels=[0, 1, 2])

    assert result["roc_auc_ovr"] is None
    assert "roc_auc_ovr" in caplog.text
