from pathlib import Path

from src.external.model_freezing import sha256_file


def test_file_hash_is_repeatable(tmp_path: Path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"frozen")
    first = sha256_file(path)
    assert first == sha256_file(path)
    path.write_bytes(b"changed")
    assert sha256_file(path) != first
