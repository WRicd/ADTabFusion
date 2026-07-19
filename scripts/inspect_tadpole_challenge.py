from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adni_inventory import scan_raw_directory, write_inventory_outputs
from src.tadpole.dictionary_parser import load_tadpole_dictionary, match_dictionary_columns
from src.tadpole.modality_mapper import infer_modality


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the full TADPOLE challenge files.")
    parser.add_argument("--data-dir", default="data/tadpole_challenge")
    parser.add_argument("--output-md", default="docs/tadpole_challenge_inventory.md")
    parser.add_argument(
        "--output-json", default="outputs/metrics/tadpole_challenge_file_inventory.json"
    )
    parser.add_argument(
        "--availability-json",
        default="outputs/metrics/tadpole_full_modality_availability.json",
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    data_path = root / "TADPOLE_D1_D2.csv"
    dictionary_path = root / "TADPOLE_D1_D2_Dict.csv"
    for path in (data_path, dictionary_path):
        if not path.exists():
            parser.error(f"Required TADPOLE file does not exist: {path}")

    inventory = scan_raw_directory(root)
    write_inventory_outputs(
        inventory,
        raw_dir=root,
        output_md=args.output_md,
        output_json=args.output_json,
        availability_json=args.availability_json,
    )
    columns = pd.read_csv(data_path, nrows=0).columns.astype(str).tolist()
    dictionary = load_tadpole_dictionary(dictionary_path)
    matches = match_dictionary_columns(columns, dictionary)
    modalities: dict[str, list[str]] = {}
    for column in columns:
        modality = infer_modality(column, matches[column])
        modalities.setdefault(modality, []).append(column)

    availability = {
        modality: {
            "available": bool(matched_columns),
            "feature_count": len(matched_columns),
            "matched_columns": matched_columns,
        }
        for modality, matched_columns in sorted(modalities.items())
    }
    output_path = Path(args.availability_json)
    output_path.write_text(
        json.dumps(availability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _append_modality_summary(Path(args.output_md), availability)
    print(f"Scanned {len(inventory)} CSV files.")
    print(f"D1/D2 columns classified: {len(columns)}")
    print(f"CSF available: {availability.get('csf', {}).get('available', False)}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.availability_json}")
    return 0


def _append_modality_summary(report_path: Path, availability: dict) -> None:
    lines = [
        "",
        "## Dictionary-backed D1/D2 modality classification",
        "",
        "| Modality | Available | Columns |",
        "|---|---|---:|",
    ]
    for modality, entry in availability.items():
        lines.append(
            f"| {modality} | {'yes' if entry['available'] else 'no'} | {entry['feature_count']} |"
        )
    lines.append("")
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())

