from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.schema_alignment import audit_d3_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen D1/D2 features against D3 without D4.")
    parser.add_argument("--config", default="configs/phase_c_direct_transfer.yaml")
    parser.add_argument("--train-csv")
    parser.add_argument("--d3-csv")
    parser.add_argument("--whitelist")
    parser.add_argument("--output", default="outputs/phase_c/audit/d3_schema_report.md")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    result = audit_d3_schema(
        args.train_csv or data["train_csv"],
        args.d3_csv or data["d3_csv"],
        args.whitelist or data["whitelist"],
        data["feature_catalog"],
        Path(args.output).parent,
        config["deployment"].get("full_feature_threshold", 0.95),
        config["deployment"].get("required_modalities"),
    )
    decision = result["decision"]
    print(f"D3 schema audit complete: {args.output}")
    print(f"Primary deployment frozen as: {decision['primary_candidate']}")


if __name__ == "__main__":
    main()
