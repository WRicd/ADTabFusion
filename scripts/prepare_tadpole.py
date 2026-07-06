from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_output_dirs, load_config
from src.data_loader import audit_dataframe, load_tadpole_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = config["project"].get("output_dir", "outputs")
    ensure_output_dirs(output_dir)
    df = load_tadpole_csv(config["data"]["raw_csv"])
    audit = audit_dataframe(
        df,
        output_dir,
        subject_col=config["data"].get("subject_col", "RID"),
        visit_col=config["data"].get("visit_col", "VISCODE"),
        label_col=config["data"].get("label_col", "DX"),
        high_missing_threshold=config.get("preprocessing", {}).get("max_missing_rate", 0.6),
    )
    print(f"Prepared audit for {audit['n_rows']} rows and {audit['n_subjects']} subjects.")


if __name__ == "__main__":
    main()

