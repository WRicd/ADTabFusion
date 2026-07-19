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
from src.tasks.future_diagnosis import (
    assert_pair_integrity,
    build_split_pairs,
    load_longitudinal_source,
    split_subjects_before_pairing,
)


def build_from_config(config: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data = config["data"]
    manifest = json.loads(Path(data["feature_source_manifest"]).read_text(encoding="utf-8"))
    features = manifest["feature_order"]
    frame = load_longitudinal_source(
        data["train_csv"], features, data.get("subject_col", "RID"),
        data.get("date_col", "EXAMDATE"), data.get("label_col", "DX"),
    )
    split_cfg = config.get("split", {})
    split = split_subjects_before_pairing(
        frame, data.get("subject_col", "RID"), config["project"].get("random_state", 42),
        split_cfg.get("val_size", 0.1), split_cfg.get("test_size", 0.2),
    )
    pair_cfg = config.get("pairing", {})
    pairs = build_split_pairs(
        frame, features, split,
        subject_col=data.get("subject_col", "RID"), date_col=data.get("date_col", "EXAMDATE"),
        label_col=data.get("label_col", "DX"),
        min_horizon_months=pair_cfg.get("min_horizon_months", 6),
        max_horizon_months=pair_cfg.get("max_horizon_months", 60),
    )
    assert_pair_integrity(pairs, features)
    combined = pd.concat(pairs.values(), ignore_index=True)
    return combined, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe D1/D2 future diagnosis pairs.")
    parser.add_argument("--config", default="configs/phase_c_horizon_aware.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    combined, pairs = build_from_config(config)
    output = Path(config["project"]["output_dir"])
    (output / "audit").mkdir(parents=True, exist_ok=True)
    (output / "evaluation").mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "audit" / "future_diagnosis_pairs.csv", index=False)
    summary = []
    for name, frame in pairs.items():
        summary.append(
            {
                "split": name,
                "subjects": int(frame["RID"].nunique()),
                "pairs": int(len(frame)),
                "pairs_per_subject_mean": float(frame.groupby("RID").size().mean()),
                "CN": int((frame["target_diagnosis"] == "CN").sum()),
                "MCI": int((frame["target_diagnosis"] == "MCI").sum()),
                "AD": int((frame["target_diagnosis"] == "AD").sum()),
            }
        )
    pd.DataFrame(summary).to_csv(output / "evaluation" / "future_pair_summary.csv", index=False)
    print(f"Built {len(combined)} future-diagnosis pairs across {combined['RID'].nunique()} subjects.")


if __name__ == "__main__":
    main()
