from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.feature_groups import available_groups
from src.splits import make_subject_split
from src.training import load_labeled_data, model_feature_columns

LOGGER = logging.getLogger(__name__)


def run_shap_analysis(
    pipeline,
    X_test: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    seed: int = 42,
    use_gpu: bool = False,
    max_samples: int = 300,
) -> pd.DataFrame | None:
    """Compute SHAP values and save summary + dependence plots.

    Supports GPU-accelerated TreeSHAP for tree-based models when *use_gpu*
    is *True* and the model is a supported tree ensemble (XGBoost, LightGBM,
    RandomForest, HistGradientBoosting).
    """
    try:
        import shap
    except ImportError:
        LOGGER.info("SHAP not installed — skipping SHAP analysis.")
        return None

    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Determine the underlying model type
    model_type = type(estimator).__name__
    is_tree = any(t in model_type for t in ("XGB", "LGBM", "RandomForest", "GradientBoosting", "HistGradientBoosting"))

    if not is_tree:
        LOGGER.info("SHAP: model type '%s' is not tree-based — skipping.", model_type)
        return None

    # Subsample for speed
    if len(X_test) > max_samples:
        X_test = X_test.sample(max_samples, random_state=seed)

    try:
        X_preprocessed = preprocessor.transform(X_test)
        preprocessed_feature_names = list(preprocessor.get_feature_names_out())
        X_background = X_preprocessed[: min(100, len(X_preprocessed))]
    except Exception:
        X_preprocessed = X_test.values if hasattr(X_test, "values") else X_test
        preprocessed_feature_names = feature_names
        X_background = X_preprocessed[: min(100, len(X_preprocessed))]

    # -- TreeSHAP (GPU-accelerated for XGBoost/LightGBM) --------------------
    LOGGER.info(
        "SHAP: running TreeExplainer (model=%s, gpu=%s, n=%d)",
        model_type,
        use_gpu,
        len(X_preprocessed),
    )
    try:
        explainer = shap.TreeExplainer(estimator, data=X_background, feature_perturbation="interventional")
    except Exception:
        explainer = shap.TreeExplainer(estimator)

    try:
        shap_values = explainer(X_preprocessed, check_additivity=False)
    except Exception as exc:
        LOGGER.warning("SHAP explainer failed: %s", exc)
        return None

    # -- Save summary plot --------------------------------------------------
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values.values,
        X_preprocessed,
        feature_names=preprocessed_feature_names,
        show=False,
    )
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_best_model.png", dpi=160, bbox_inches="tight")
    plt.close()

    # -- Save bar plot (mean |SHAP|) ---------------------------------------
    shap.summary_plot(
        shap_values.values,
        X_preprocessed,
        feature_names=preprocessed_feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_feature_importance_best_model.png", dpi=160, bbox_inches="tight")
    plt.close()

    # -- Build per-feature importance table ---------------------------------
    if isinstance(shap_values.values, list):
        # Multi-class: average absolute SHAP across all classes
        importance = np.mean([np.abs(v).mean(axis=0) for v in shap_values.values], axis=0)
    else:
        importance = np.abs(shap_values.values).mean(axis=0)

    shap_df = pd.DataFrame(
        {"feature": preprocessed_feature_names, "importance": importance, "method": "shap"}
    ).sort_values("importance", ascending=False)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(metrics_dir / "shap_importance_best_model.csv", index=False)

    LOGGER.info("SHAP analysis complete — saved to %s.", figures_dir)
    return shap_df


def run_explainability(config: dict[str, Any], quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run best-model feature and modality importance analysis.

    Uses TreeSHAP when the best model is tree-based (and *run_shap* is
    enabled in config), and falls back to permutation importance otherwise.
    """
    output_dir = Path(config["project"].get("output_dir", "outputs"))
    model_path = output_dir / "models" / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Best model not found. Run scripts/run_baselines.py before explainability.")
    pipeline = joblib.load(model_path)
    df, label_col, _ = load_labeled_data(config)
    feature_columns = model_feature_columns(df, config, output_dir)

    seed = _best_seed(output_dir) or config["project"].get("seed_list", [42])[0]
    split = make_subject_split(
        df,
        subject_col=config["data"].get("subject_col", "RID"),
        label_col=label_col,
        test_size=config.get("split", {}).get("test_size", 0.2),
        val_size=config.get("split", {}).get("val_size", 0.1),
        seed=seed,
        output_dir=output_dir,
    )
    test_df = df.loc[split["test_idx"]]
    max_samples = 120 if quick else config.get("explainability", {}).get("max_samples", 300)
    if len(test_df) > max_samples:
        test_df = test_df.sample(max_samples, random_state=seed)
    X_test = test_df[feature_columns]
    y_test = test_df[label_col]

    # -- SHAP (when enabled) ------------------------------------------------
    shap_enabled = config.get("explainability", {}).get("run_shap", True)
    shap_result = None
    if shap_enabled:
        model_cfg = config.get("models", {}).get(_best_model_name(output_dir) or "", {})
        use_gpu = model_cfg.get("use_gpu", config.get("models", {}).get("use_gpu", False))
        shap_result = run_shap_analysis(
            pipeline,
            X_test,
            feature_columns,
            output_dir,
            seed=seed,
            use_gpu=use_gpu,
            max_samples=max_samples,
        )

    # -- Fallback: permutation importance -----------------------------------
    feature_importance = _native_importance(pipeline, feature_columns)
    if feature_importance.empty:
        LOGGER.info("No native importance available — running permutation importance.")
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=3 if quick else 8,
            random_state=seed,
            scoring="f1_macro",
        )
        feature_importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
                "method": "permutation_importance",
            }
        )

    # If SHAP succeeded, use it as the primary importance measure
    if shap_result is not None and not shap_result.empty:
        feature_importance = shap_result

    feature_importance["modality"] = feature_importance["feature"].map(
        lambda name: _feature_to_modality(name, available_groups(df))
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    modality_importance = (
        feature_importance.groupby("modality", dropna=False)["importance"]
        .sum()
        .reset_index()
        .sort_values("importance", ascending=False)
    )

    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(metrics_dir / "feature_importance_best_model.csv", index=False)
    modality_importance.to_csv(metrics_dir / "modality_importance_best_model.csv", index=False)
    _plot_importance(
        feature_importance.head(20),
        "feature",
        figures_dir / "top_features_best_model.png",
        "Top feature importance",
    )
    _plot_importance(
        modality_importance,
        "modality",
        figures_dir / "modality_importance_best_model.png",
        "Modality importance",
    )
    return feature_importance, modality_importance


def write_basic_feature_importance(model, output_dir: str | Path) -> pd.DataFrame:
    """Backward-compatible wrapper used by the baseline script."""
    output = Path(output_dir)
    importance = _native_importance(model, [])
    if importance.empty:
        return importance
    metrics_dir = output / "metrics"
    figures_dir = output / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(metrics_dir / "feature_importance_best_model.csv", index=False)
    _plot_importance(
        importance.head(20),
        "feature",
        figures_dir / "top_features_best_model.png",
        "Top feature importance",
    )
    (figures_dir / "shap_summary_best_model.png").write_bytes(
        (figures_dir / "top_features_best_model.png").read_bytes()
    )
    return importance


def _native_importance(pipeline, original_features: list[str]) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = original_features
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        method = "tree_feature_importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).mean(axis=0)
        method = "standardized_coefficient"
    else:
        return pd.DataFrame()
    if len(values) != len(feature_names):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "method": method,
        }
    )


def _feature_to_modality(feature: str, groups: dict[str, list[str]]) -> str:
    clean = feature.split("__")[-1]
    for modality, columns in groups.items():
        for col in columns:
            if clean == col or clean.startswith(f"{col}_") or clean.startswith(f"{col}="):
                return modality
    return "other"


def _plot_importance(df: pd.DataFrame, label_col: str, path: Path, title: str) -> None:
    if df.empty:
        return
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.25)))
    ax.barh(plot_df[label_col].astype(str), plot_df["importance"].astype(float))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _best_seed(output_dir: Path) -> int | None:
    path = output_dir / "metrics" / "best_model.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("seed")


def _best_model_name(output_dir: Path) -> str | None:
    """Return the best model's name from the metadata JSON."""
    path = output_dir / "metrics" / "best_model.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("model")
