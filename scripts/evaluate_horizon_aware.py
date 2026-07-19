from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.bootstrap import subject_bootstrap_ci
from src.external.external_evaluation import CLASS_TO_ID, evaluate_external_predictions, first_follow_up
from src.external.inference import probability_frame
from src.external.schema_alignment import align_to_frozen_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and evaluate frozen horizon-aware D3/D4 inference.")
    parser.add_argument("--config", default="configs/phase_c_external_evaluation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["project"]["output_dir"])
    frame = pd.read_csv(root / "evaluation" / "d3_d4_matched_rows.csv", low_memory=False)
    manifest = json.loads((root / "manifests" / "horizon_aware_manifest.json").read_text(encoding="utf-8"))
    model = joblib.load(root / "models" / "horizon_aware_pipeline.joblib")
    aligned = align_to_frozen_schema(frame, manifest["feature_order"])
    predicted = probability_frame(model.predict_proba(aligned))
    predicted.insert(0, "forecast_months", frame["forecast_months"].to_numpy())
    predicted.insert(0, "D4_SCANDATE", frame["D4_SCANDATE"].to_numpy())
    predicted.insert(0, "D3_EXAMDATE", frame["D3_EXAMDATE"].to_numpy())
    predicted.insert(0, "RID", frame["RID"].to_numpy())
    predicted.insert(0, "evaluation_row_id", frame["evaluation_row_id"].to_numpy())
    predicted["model_id"] = manifest["model_id"]
    predicted.to_csv(root / "predictions" / "d3_d4_horizon_aware.csv", index=False)
    evaluation = frame.copy()
    for column in ["predicted_class", "prob_CN", "prob_MCI", "prob_AD"]:
        evaluation[column] = predicted[column].to_numpy()
    metrics = evaluate_external_predictions(
        evaluation, "predicted_class", ["prob_CN", "prob_MCI", "prob_AD"],
        manifest["model_id"], config.get("minimum_stable_rows", 20),
    )
    metrics.to_csv(root / "evaluation" / "horizon_aware_metrics.csv", index=False)
    first = first_follow_up(evaluation)
    first["_predicted_label"] = first["predicted_class"].map(CLASS_TO_ID)
    boot = config.get("bootstrap", {})
    ci = subject_bootstrap_ci(
        first, "D4_label", "_predicted_label", ["prob_CN", "prob_MCI", "prob_AD"],
        repetitions=boot.get("repetitions", 1000),
        confidence_level=boot.get("confidence_level", 0.95), seed=boot.get("seed", 42),
    )
    ci.insert(0, "scope", "first_follow_up")
    ci.insert(0, "model_id", manifest["model_id"])
    ci.to_csv(root / "evaluation" / "horizon_aware_bootstrap_ci.csv", index=False)
    direct_ci = pd.read_csv(root / "evaluation" / "direct_transfer_bootstrap_ci.csv")
    pd.concat([direct_ci, ci], ignore_index=True).to_csv(root / "evaluation" / "bootstrap_ci.csv", index=False)
    print(f"Evaluated horizon-aware predictions on {len(frame)} matched rows.")


if __name__ == "__main__":
    main()
