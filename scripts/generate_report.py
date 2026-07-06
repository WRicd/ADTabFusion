from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/reports/final_report.md")
    args = parser.parse_args()
    generate_report(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

