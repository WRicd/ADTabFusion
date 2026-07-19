from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tadpole.feature_catalog import (
    build_feature_catalog,
    catalog_summary,
    write_feature_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the full TADPOLE feature catalog.")
    parser.add_argument("--data", default="data/tadpole_challenge/TADPOLE_D1_D2.csv")
    parser.add_argument(
        "--dictionary", default="data/tadpole_challenge/TADPOLE_D1_D2_Dict.csv"
    )
    parser.add_argument("--d3", default="data/tadpole_challenge/TADPOLE_D3.csv")
    parser.add_argument(
        "--output-csv", default="outputs/metrics/tadpole_full_feature_catalog.csv"
    )
    parser.add_argument("--output-md", default="docs/tadpole_full_feature_catalog.md")
    parser.add_argument(
        "--modality-map-md", default="docs/tadpole_full_modality_map.md"
    )
    parser.add_argument("--max-missing-rate", type=float, default=0.70)
    parser.add_argument("--min-non-missing-count", type=int, default=100)
    parser.add_argument("--max-features-per-family", type=int, default=1)
    parser.add_argument("--max-primary-features", type=int, default=200)
    args = parser.parse_args()

    catalog = build_feature_catalog(
        args.data,
        args.dictionary,
        d3_path=args.d3,
        max_missing_rate=args.max_missing_rate,
        min_non_missing_count=args.min_non_missing_count,
        max_features_per_family=args.max_features_per_family,
        max_primary_features=args.max_primary_features,
    )
    write_feature_catalog(
        catalog, args.output_csv, args.output_md, modality_map_md=args.modality_map_md
    )
    summary = catalog_summary(catalog)
    print(f"Audited {summary['columns']} columns.")
    print(f"Dictionary matches: {summary['dictionary_matches']}")
    print(f"Primary whitelist candidates: {summary['primary_features']}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.modality_map_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
