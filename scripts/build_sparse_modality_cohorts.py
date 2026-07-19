from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.tadpole.phase_b_experiments import build_sparse_modality_cohort


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sparse TADPOLE modality cohorts.")
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()
    for config_path in args.configs:
        summary = build_sparse_modality_cohort(load_config(config_path))
        print(
            f"{summary['modality']}: {summary['available_subjects']} subjects, "
            f"{summary['available_visits']} visits"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

