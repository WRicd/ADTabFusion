from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.phase_d.temporal_split import assert_subject_split_disjoint, build_temporal_subject_split, write_temporal_split


def main() -> None:
    config = load_config("configs/phase_d_d3_core.yaml")
    data = config["data"]
    frame = pd.read_csv(data["train_csv"], usecols=[data["subject_col"], data["date_col"]], low_memory=False)
    split, assignment = build_temporal_subject_split(frame, data["subject_col"], data["date_col"])
    assert_subject_split_disjoint(split)
    write_temporal_split(split, assignment, "outputs/phase_d/cohorts")
    print("Locked temporal split: " + ", ".join(f"{name}={len(ids)}" for name, ids in split.items()))


if __name__ == "__main__": main()
