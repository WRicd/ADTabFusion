from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.d3_cohort import select_d3_index_records
from src.external.inference import predict_with_frozen_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict D3 with frozen direct-transfer models.")
    parser.add_argument("--config", default="configs/phase_c_direct_transfer.yaml")
    parser.add_argument("--model")
    parser.add_argument("--d3")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    root = Path(config["project"]["output_dir"])
    roles = ["primary"] if args.model or args.output else ["primary", "sensitivity"]
    manifests = {}
    all_features = []
    for role in roles:
        manifest_path = root / "manifests" / f"{role}_model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[role] = manifest_path
        all_features.extend(manifest["feature_order"])
    subject_col = data.get("subject_col", "RID")
    date_col = data.get("date_col", "EXAMDATE")
    required = list(dict.fromkeys([subject_col, date_col, *all_features]))
    raw = pd.read_csv(
        args.d3 or data["d3_csv"],
        usecols=lambda column: column in required,
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    cohort, _ = select_d3_index_records(raw, subject_col, date_col)
    for role in roles:
        model = Path(args.model) if args.model else root / "models" / f"{role}_pipeline.joblib"
        manifest = manifests[role]
        predicted, missing = predict_with_frozen_pipeline(model, manifest, cohort)
        output = Path(args.output) if args.output else root / "predictions" / f"d3_direct_{role}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        identity = pd.DataFrame(
            {
                "RID": cohort[subject_col].to_numpy(),
                "D3_EXAMDATE": cohort[date_col].to_numpy(),
            }
        )
        pd.concat([identity, predicted], axis=1).to_csv(output, index=False)
        print(f"Wrote {len(predicted)} predictions to {output}")
        if missing:
            print(f"WARNING: {len(missing)} frozen features absent in D3 and padded with NaN.")


if __name__ == "__main__":
    main()
