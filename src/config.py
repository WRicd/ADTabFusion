from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create standard output folders and return their paths."""
    root = Path(output_dir)
    paths = {
        "root": root,
        "figures": root / "figures",
        "metrics": root / "metrics",
        "models": root / "models",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

