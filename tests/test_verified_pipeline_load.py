from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.external.model_freezing import load_verified_pipeline, sha256_file


def test_pipeline_loads_when_digest_matches_manifest(tmp_path: Path):
    path = tmp_path / "pipeline.joblib"
    joblib.dump(LogisticRegression(), path)
    restored = load_verified_pipeline(path, sha256_file(path))
    assert isinstance(restored, LogisticRegression)


def test_tampered_pipeline_is_never_deserialized(tmp_path: Path):
    path = tmp_path / "pipeline.joblib"
    joblib.dump(LogisticRegression(), path)
    digest = sha256_file(path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="Refusing to load"):
        load_verified_pipeline(path, digest)
