from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.tadpole.leakage_rules import write_leakage_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit full TADPOLE feature leakage.")
    parser.add_argument("--config", default="configs/tadpole_full_internal.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config.get("paths", {})
    catalog_path = Path(
        paths.get("feature_catalog", "outputs/metrics/tadpole_full_feature_catalog.csv")
    )
    if not catalog_path.exists():
        parser.error(
            f"Feature catalog does not exist: {catalog_path}. Run build_tadpole_feature_catalog.py first."
        )
    catalog = pd.read_csv(catalog_path)
    blacklist, whitelist = write_leakage_artifacts(
        catalog,
        paths.get("feature_blacklist", "outputs/metrics/tadpole_full_feature_blacklist.json"),
        paths.get("feature_whitelist", "outputs/metrics/tadpole_full_feature_whitelist.json"),
        paths.get("leakage_report", "docs/tadpole_full_leakage_audit.md"),
    )
    print(f"Blacklist columns: {len(blacklist)}")
    print(f"Primary whitelist columns: {len(whitelist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

