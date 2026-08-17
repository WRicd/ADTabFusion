from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType


def agg_pyplot(figure_dir: str | Path) -> ModuleType:
    """Return ``matplotlib.pyplot`` configured for headless figure export.

    MPLCONFIGDIR must point at a writable directory and the Agg backend must be
    selected before pyplot is imported, otherwise figure generation fails on
    machines without a display or a home directory cache.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(Path(figure_dir) / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def optional_agg_pyplot(figure_dir: str | Path) -> ModuleType | None:
    """Like :func:`agg_pyplot`, but return ``None`` when matplotlib is absent."""
    try:
        return agg_pyplot(figure_dir)
    except ImportError:
        return None
