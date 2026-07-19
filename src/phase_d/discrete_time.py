from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.phase_d.mci_conversion import binary_metrics, select_mci_index_visits, _fit_binary
from src.external.model_freezing import sha256_file


def build_person_period_data(index_visits: pd.DataFrame, interval_ends: list[int]) -> pd.DataFrame:
    rows = []
    for _, subject in index_visits.iterrows():
        previous = 0.0
        for interval_index, end in enumerate(interval_ends, 1):
            event_month = subject["first_ad_months"]
            event = bool(pd.notna(event_month) and previous < event_month <= end)
            observed = event or subject["max_followup_months"] >= end
            if not observed:
                break
            row = subject.to_dict()
            row.update({"interval_index": interval_index, "interval_start_months": previous, "elapsed_months": end, "event": int(event), "label": int(event)})
            rows.append(row)
            if event:
                break
            previous = end
    periods = pd.DataFrame(rows)
    if not periods.empty:
        counts = periods.groupby("RID")["RID"].transform("size")
        periods["subject_weight"] = 1.0 / counts
    return periods


def train_discrete_time_model(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["project"]["output_dir"])
    data = config["data"]
    features = json.loads(Path(data["feature_profile"]).read_text(encoding="utf-8"))
    required = [data.get("subject_col", "RID"), data.get("date_col", "EXAMDATE"), data.get("label_col", "DX"), *features]
    frame = pd.read_csv(data["train_csv"], usecols=lambda column: column in required, low_memory=False, na_values=["", " ", "-4", "-4.0"])
    index = select_mci_index_visits(frame, features, data.get("subject_col", "RID"), data.get("date_col", "EXAMDATE"), data.get("label_col", "DX"))
    split = json.loads(Path(data["temporal_split"]).read_text(encoding="utf-8"))["splits"]
    rid_to_split = {str(rid): name for name, ids in split.items() for rid in ids}
    index["split"] = index["RID"].astype(str).map(rid_to_split)
    periods = build_person_period_data(index, config["intervals"])
    model_features = [*features, "interval_index", "elapsed_months"]
    train, validation, test = (periods[periods["split"] == name].copy() for name in ("train", "validation", "temporal_test"))
    rows, candidates = [], {}
    for model_name in config["models"]["run"]:
        pipeline = _fit_binary(train, model_features, model_name, config)
        candidates[model_name] = pipeline
        metrics = binary_metrics(validation["label"].to_numpy(), pipeline.predict_proba(validation[model_features])[:, 1])
        rows.append({"model": model_name, "split": "validation", "n_rows": len(validation), **metrics})
    results = pd.DataFrame(rows)
    selected = results.sort_values(["log_loss", "pr_auc"], ascending=[True, False]).iloc[0]
    name = str(selected["model"]); pipeline = candidates[name]
    test_metrics = binary_metrics(test["label"].to_numpy(), pipeline.predict_proba(test[model_features])[:, 1])
    pd.DataFrame([{ "model": name, "split": "locked_temporal_test", "n_rows": len(test), **test_metrics}]).to_csv(output / "temporal_validation" / "mci_discrete_time_metrics.csv", index=False)
    results.to_csv(output / "internal_validation" / "mci_discrete_time_validation.csv", index=False)
    model_path = output / "models" / "mci_discrete_time_pipeline.joblib"; joblib.dump(pipeline, model_path)
    manifest = {"model_id": f"phase_d_mci_discrete_{name}", "model_name": name, "feature_order": model_features, "interval_ends_months": config["intervals"], "d4_used_for_selection": False, "model_sha256": sha256_file(model_path)}
    (output / "manifests" / "mci_discrete_time_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    risk = predict_cumulative_risk(pipeline, index[index["split"] == "temporal_test"], features, config["intervals"])
    risk.to_csv(output / "temporal_validation" / "mci_cumulative_risk_predictions.csv", index=False)
    _plot_risk(risk, output / "figures" / "mci_risk_by_horizon.png")
    return {"manifest": manifest, "metrics": test_metrics}


def predict_cumulative_risk(pipeline, index_visits: pd.DataFrame, features: list[str], interval_ends: list[int]) -> pd.DataFrame:
    rows = []
    for _, subject in index_visits.iterrows():
        cumulative_survival = 1.0
        for interval_index, end in enumerate(interval_ends, 1):
            row = {feature: subject[feature] for feature in features}
            row.update({"interval_index": interval_index, "elapsed_months": end})
            hazard = float(pipeline.predict_proba(pd.DataFrame([row]))[0, 1])
            cumulative_survival *= 1.0 - hazard
            rows.append({"RID": subject["RID"], "horizon_months": end, "hazard": hazard, "cumulative_risk": 1.0 - cumulative_survival})
    return pd.DataFrame(rows)


def _plot_risk(risk: pd.DataFrame, path: Path) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib; matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    summary = risk.groupby("horizon_months")["cumulative_risk"].agg(["mean", "median"])
    fig, ax = plt.subplots(figsize=(12.8, 7.2)); ax.plot(summary.index, summary["mean"], marker="o", label="Mean risk"); ax.plot(summary.index, summary["median"], marker="s", label="Median risk")
    ax.set_ylim(0, 1); ax.set_xlabel("Horizon (months)"); ax.set_ylabel("Cumulative AD Conversion Risk"); ax.set_title("MCI Conversion Risk by Horizon"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
