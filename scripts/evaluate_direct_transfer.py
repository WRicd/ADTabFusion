from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.bootstrap import subject_bootstrap_ci
from src.external.external_evaluation import CLASS_TO_ID, evaluate_external_predictions, first_follow_up


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen direct-transfer D3 predictions on D4.")
    parser.add_argument("--config", default="configs/phase_c_external_evaluation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["project"]["output_dir"])
    frame = pd.read_csv(root / "evaluation" / "d3_d4_matched_rows.csv", low_memory=False)
    metric_frames = []
    ci_frames = []
    boot = config.get("bootstrap", {})
    for role in ("primary", "sensitivity"):
        prefix = f"direct_{role}_"
        prediction_col = prefix + "predicted_class"
        probability_cols = [prefix + "prob_CN", prefix + "prob_MCI", prefix + "prob_AD"]
        model_id = str(frame[prefix + "model_id"].iloc[0])
        metric_frames.append(
            evaluate_external_predictions(
                frame, prediction_col, probability_cols, model_id,
                config.get("minimum_stable_rows", 20),
            )
        )
        first = first_follow_up(frame)
        first["_predicted_label"] = first[prediction_col].map(CLASS_TO_ID)
        ci = subject_bootstrap_ci(
            first, "D4_label", "_predicted_label", probability_cols,
            repetitions=boot.get("repetitions", 1000),
            confidence_level=boot.get("confidence_level", 0.95), seed=boot.get("seed", 42),
        )
        ci.insert(0, "scope", "first_follow_up")
        ci.insert(0, "model_id", model_id)
        ci_frames.append(ci)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(root / "evaluation" / "direct_transfer_metrics.csv", index=False)
    ci = pd.concat(ci_frames, ignore_index=True)
    ci.to_csv(root / "evaluation" / "direct_transfer_bootstrap_ci.csv", index=False)
    ci.to_csv(root / "evaluation" / "bootstrap_ci.csv", index=False)
    print(f"Evaluated direct transfer on {len(frame)} matched rows and {frame['RID'].nunique()} subjects.")


if __name__ == "__main__":
    main()
