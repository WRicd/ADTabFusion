"""Guardrail: hyperparameter search must never touch the locked temporal test.

Phase D's scientific claim depends on the temporal test being a one-shot
measurement. If a search loop could read it, every reported test metric would
become an optimistically biased selection metric instead.
"""

import ast
import inspect
from pathlib import Path

import src.tuning as tuning

ROOT = Path(__file__).resolve().parents[1]
TUNING_SCRIPT = ROOT / "scripts" / "tune_hyperparameters.py"


def test_tuning_api_exposes_no_test_split_parameter():
    parameters = inspect.signature(tuning.tune_model).parameters
    assert not [name for name in parameters if "test" in name.lower()]


def test_tuning_script_never_materializes_the_temporal_test_split():
    """The script may name the split only to document that it is excluded."""
    source = TUNING_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Any comparison against the temporal_test split would select those rows.
    offending = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == "temporal_test"
    ]
    assert not offending, "tune_hyperparameters.py must not select temporal_test rows"


def test_tuning_script_only_fits_on_train_and_validation():
    source = TUNING_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    split_literals = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    used_splits = split_literals & {"train", "validation", "temporal_test", "excluded"}

    assert used_splits <= {"train", "validation"}


def test_selection_metrics_are_computed_from_validation_only():
    """_score is the single scoring path; it takes no split argument."""
    parameters = inspect.signature(tuning._score).parameters
    assert list(parameters) == ["y_true", "y_proba", "metric", "labels"]
