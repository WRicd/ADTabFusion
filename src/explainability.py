from __future__ import annotations

import json
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


def run_explainability(config: dict[str, Any], quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run best-model feature and modality importance analysis."""
    output_dir = Path(config["project"].get("output_dir", "outputs"))
    model_path = output_dir / "models" / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "Best model not found. Run scripts/run_baselines.py before explainability."
        )
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

    feature_importance = _native_importance(pipeline, feature_columns)
    if feature_importance.empty:
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

