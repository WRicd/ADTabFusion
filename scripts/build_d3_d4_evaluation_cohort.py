from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.d4_matching import build_d3_d4_evaluation_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description="Match frozen D3 records and predictions to D4 outcomes.")
    parser.add_argument("--config", default="configs/phase_c_external_evaluation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    manifest = json.loads(Path(data["horizon_manifest"]).read_text(encoding="utf-8"))
    _, summary = build_d3_d4_evaluation_cohort(
        data["d3_csv"], data["d4_csv"], data["direct_primary_predictions"],
        data["direct_sensitivity_predictions"], manifest["base_feature_order"],
        Path(config["project"]["output_dir"]) / "evaluation",
        data.get("subject_col", "RID"), data.get("d3_date_col", "EXAMDATE"),
        data.get("d4_date_col", "ScanDate"), data.get("d4_label_col", "Diagnosis"),
    )
    print(f"Matched {summary['matched_rows']} D4 rows from {summary['matched_subjects']} subjects.")


if __name__ == "__main__":
    main()
