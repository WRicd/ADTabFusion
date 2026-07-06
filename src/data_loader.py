from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def load_tadpole_csv(path: str | Path) -> pd.DataFrame:
    """Load TADPOLE CSV and normalize column name whitespace."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            "Please download the ADNI/TADPOLE-derived CSV according to the data "
            "access agreement and place it at the expected path."
        )
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def audit_dataframe(
    df: pd.DataFrame,
    output_dir: str | Path,
    subject_col: str = "RID",
    visit_col: str = "VISCODE",
    label_col: str = "DX",
    high_missing_threshold: float = 0.60,
) -> dict[str, Any]:
    """Write a compact data audit JSON and simple diagnostic figures."""
    output = Path(output_dir)
    report_dir = output / "reports"
    figure_dir = output / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    missing_rate = df.isna().mean().sort_values(ascending=False)
    label_distribution = (
        df[label_col].fillna("<missing>").astype(str).value_counts().to_dict()
        if label_col in df.columns
        else {}
    )
    duplicate_subject_visit = 0
    if subject_col in df.columns and visit_col in df.columns:
        duplicate_subject_visit = int(df.duplicated([subject_col, visit_col]).sum())

    audit = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_subjects": int(df[subject_col].nunique()) if subject_col in df.columns else None,
        "label_distribution": label_distribution,
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
        "categorical_columns": int(df.select_dtypes(exclude="number").shape[1]),
        "high_missing_columns": missing_rate[
            missing_rate > high_missing_threshold
        ].index.tolist(),
        "missing_rate_by_column": {
            col: float(rate) for col, rate in missing_rate.items()
        },
        "duplicate_subject_visit_rows": duplicate_subject_visit,
    }
    (report_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_missing_rate(missing_rate, figure_dir / "missing_rate_by_column.png")
    _plot_label_distribution(label_distribution, figure_dir / "label_distribution.png")
    return audit


def _plot_missing_rate(missing_rate: pd.Series, path: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    top = missing_rate.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.25)))
    ax.barh(top.index.astype(str), top.values)
    ax.set_xlabel("Missing rate")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_label_distribution(labels: dict[str, int], path: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(labels.keys()), list(labels.values()))
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
