from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.phase_d.transition_model import train_transition_aware


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase_d_transition_model.yaml")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-fit progress logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    result = train_transition_aware(load_config(args.config), args.config)
    print(
        f"Selected {result['manifest']['model_name']}; "
        f"locked temporal Macro F1={result['temporal_test']['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
