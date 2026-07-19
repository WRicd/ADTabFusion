from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adni_inventory import scan_raw_directory, write_inventory_outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect local ADNI CSV structure and infer available modalities."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-md", default="docs/adni_file_inventory.md")
    parser.add_argument(
        "--output-json", default="outputs/metrics/adni_file_inventory.json"
    )
    parser.add_argument(
        "--availability-json",
        default="outputs/metrics/adni_modality_availability.json",
    )
    args = parser.parse_args()

    try:
        inventory = scan_raw_directory(args.raw_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))

    availability = write_inventory_outputs(
        inventory,
        raw_dir=args.raw_dir,
        output_md=args.output_md,
        output_json=args.output_json,
        availability_json=args.availability_json,
    )
    readable = sum(item["read_status"] != "failed" for item in inventory)
    available = [key for key, value in availability.items() if value["available"]]
    missing = [key for key, value in availability.items() if not value["available"]]
    print(f"Scanned {len(inventory)} CSV files ({readable} readable).")
    print(f"Available categories: {', '.join(available) if available else 'none'}")
    print(f"Missing categories: {', '.join(missing) if missing else 'none'}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.availability_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
