from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Hash a file in blocks so large frozen artifacts never load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_subject_hash(subjects: pd.Series) -> str:
    """Hash a subject identifier column independently of row order."""
    values = sorted({str(value) for value in subjects.dropna()})
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def git_commit_hash() -> str | None:
    """Return the current commit for model manifests, or None outside a checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
