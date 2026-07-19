from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adni_inventory import next_download_recommendations


def _status(availability: dict, key: str) -> str:
    return "OK" if availability.get(key, {}).get("available", False) else "MISSING"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check required and recommended ADNI table categories."
    )
    parser.add_argument(
        "--inventory", default="outputs/metrics/adni_modality_availability.json"
    )
    args = parser.parse_args()
    path = Path(args.inventory)
    if not path.exists():
        parser.error(f"Modality availability file does not exist: {path}")
    availability = json.loads(path.read_text(encoding="utf-8"))

    print("Required first-batch categories:")
    for label, key in [
        ("data_dictionary", "data_dictionary"),
        ("diagnosis", "diagnosis"),
        ("demographics", "demographic"),
        ("apoe", "apoe"),
        ("cognitive", "cognitive"),
    ]:
        print(f"[{_status(availability, key)}] {label}")

    print("\nRecommended second-batch categories:")
    for key in ["mri_measurement", "pet_measurement", "biofluid_csf"]:
        print(f"[{_status(availability, key)}] {key}")

    recommendations = next_download_recommendations(availability)
    print("\nNext action:")
    if recommendations:
        for recommendation in recommendations:
            print(f"- {recommendation}")
    else:
        print("- No missing second-batch modality category was detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
