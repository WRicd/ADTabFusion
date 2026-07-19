from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.d3_cohort import prepare_d3_cohort
from src.tadpole.phase_b import load_catalog_groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one D3 index record per RID without D4.")
    parser.add_argument("--config", default="configs/phase_c_direct_transfer.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    features = json.loads(Path(data["whitelist"]).read_text(encoding="utf-8"))
    groups = load_catalog_groups(data["feature_catalog"], features)
    _, summary = prepare_d3_cohort(
        data["d3_csv"],
        Path(config["project"]["output_dir"]) / "audit",
        features,
        groups,
        data.get("subject_col", "RID"),
        data.get("date_col", "EXAMDATE"),
    )
    print(f"Prepared {summary['selected_rows']} D3 inference subjects.")


if __name__ == "__main__":
    main()
