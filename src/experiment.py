from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

_ACTIVE_RUN = None


def is_available() -> bool:
    """Check whether MLflow is installed."""
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def configure_tracking(
    tracking_uri: str | None = None,
    experiment_name: str = "AD-TabFusion",
    nested: bool = True,
) -> None:
    """One-time MLflow setup.  Safe to call repeatedly."""
    if not is_available():
        LOGGER.info("MLflow not installed — experiment tracking disabled.")
        return

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Default to local filesystem under outputs/
        uri = f"file:{Path.cwd() / 'outputs' / 'mlruns'}"
        mlflow.set_tracking_uri(uri)
        LOGGER.info("MLflow tracking URI: %s", uri)

    mlflow.set_experiment(experiment_name)
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", experiment_name)
    os.environ.setdefault("MLFLOW_NESTED_RUN", "true" if nested else "false")


def start_run(
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = True,
) -> Any:
    """Start an MLflow run (no-op if MLflow unavailable)."""
    global _ACTIVE_RUN

    if not is_available():
        return None

    import mlflow

    _ACTIVE_RUN = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
    return _ACTIVE_RUN


def end_run(status: str = "FINISHED") -> None:
    """End the current MLflow run (no-op if MLflow unavailable)."""
    global _ACTIVE_RUN

    if not is_available():
        _ACTIVE_RUN = None
        return

    import mlflow

    try:
        mlflow.end_run(status=status)
    except Exception:
        pass
    _ACTIVE_RUN = None


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to the active MLflow run."""
    if not is_available():
        return
    import mlflow

    try:
        mlflow.log_params(params)
    except Exception as exc:
        LOGGER.warning("MLflow log_params failed: %s", exc)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to the active MLflow run."""
    if not is_available():
        return
    import mlflow

    try:
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None}, step=step)
    except Exception as exc:
        LOGGER.warning("MLflow log_metrics failed: %s", exc)


def log_artifact(local_path: str | Path) -> None:
    """Upload a local file / directory as an MLflow artifact."""
    if not is_available():
        return
    import mlflow

    try:
        mlflow.log_artifact(str(local_path))
    except Exception as exc:
        LOGGER.warning("MLflow log_artifact failed: %s", exc)


def log_figure(fig, artifact_path: str = "figures") -> None:
    """Log a matplotlib figure to the active MLflow run."""
    if not is_available():
        return
    import mlflow

    try:
        mlflow.log_figure(fig, artifact_path)
    except Exception as exc:
        LOGGER.warning("MLflow log_figure failed: %s", exc)


def log_dataframe(df: pd.DataFrame, artifact_path: str) -> None:
    """Persist a DataFrame as a CSV artifact under the active run."""
    if not is_available():
        return
    import mlflow

    try:
        with mlflow.start_run(run_id=_get_active_run_id(), nested=False):
            mlflow.log_text(df.to_csv(index=False), artifact_path)
    except Exception as exc:
        LOGGER.warning("MLflow log_text failed: %s", exc)


def log_model(model: Any, artifact_path: str = "model") -> None:
    """Log a fitted sklearn pipeline as an MLflow model artifact."""
    if not is_available():
        return
    try:
        import mlflow.sklearn  # noqa: F811

        with mlflow.start_run(run_id=_get_active_run_id(), nested=False):
            mlflow.sklearn.log_model(model, artifact_path)
    except Exception as exc:
        LOGGER.warning("MLflow log_model failed: %s", exc)


def set_tags(tags: dict[str, str]) -> None:
    """Set tags on the active MLflow run."""
    if not is_available():
        return
    import mlflow

    try:
        mlflow.set_tags(tags)
    except Exception as exc:
        LOGGER.warning("MLflow set_tags failed: %s", exc)


def _get_active_run_id() -> str | None:
    """Return the active run ID, or None."""
    if not is_available():
        return None
    import mlflow

    active = mlflow.active_run()
    if active is not None:
        return active.info.run_id
    return None


class ExperimentContext:
    """Context manager that wraps start/end/log_params in one block.

    Usage::

        with ExperimentContext("phase_d_xgboost", params={"lr": 0.05}) as ctx:
            ctx.log_metrics({"macro_f1": 0.881, "accuracy": 0.90})
            ctx.log_artifact("outputs/figures/roc.png")
    """

    def __init__(
        self,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}

    def __enter__(self) -> ExperimentContext:
        start_run(self.run_name)
        if self.params:
            log_params(self.params)
        if self.tags:
            set_tags(self.tags)
        return self

    def __exit__(self, *args: Any) -> None:
        end_run()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        log_artifact(path)

    def log_figure(self, fig, path: str = "figures") -> None:
        log_figure(fig, path)

    def log_model(self, model: Any, path: str = "model") -> None:
        log_model(model, path)
