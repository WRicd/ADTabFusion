from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.phase_d.d3_feature_profile import build_d3_feature_profiles


def main() -> None:
    config = load_config("configs/phase_d_d3_core.yaml")
    compact = load_config(config["data"]["compact_config"])["data"]["features"]
    data = config["data"]
    profiles = build_d3_feature_profiles(data["train_csv"], data["d3_csv"], data["whitelist"], compact, data["feature_catalog"], "outputs/phase_d/audit")
    print("Frozen D3 profiles: " + ", ".join(f"{name}={len(values)}" for name, values in profiles.items()))


if __name__ == "__main__": main()
