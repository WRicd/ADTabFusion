from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.horizon_model import train_and_freeze_horizon_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and freeze the D1/D2 horizon-aware model.")
    parser.add_argument("--config", default="configs/phase_c_horizon_aware.yaml")
    args = parser.parse_args()
    manifest = train_and_freeze_horizon_model(load_config(args.config), args.config)
    print(f"Frozen horizon-aware model: {manifest['model_name']} ({manifest['model_sha256'][:12]})")


if __name__ == "__main__":
    main()
