from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tadpole.phase_b_report import generate_phase_b_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Phase B TADPOLE report.")
    parser.add_argument(
        "--output", default="outputs/reports/phase_b_full_tadpole_report.md"
    )
    args = parser.parse_args()
    generate_phase_b_report(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

