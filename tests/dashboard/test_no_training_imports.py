import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PREFIXES = ("src.models", "src.training", "src.phase_d", "scripts.")
FORBIDDEN_CALLS = {"fit", "fit_transform", "calibrate", "dump", "to_csv", "to_json", "write_text", "write_bytes"}


def test_dashboard_has_no_training_imports_or_artifact_writes() -> None:
    active_paths = [
        ROOT / "dashboard" / "app.py",
        ROOT / "dashboard" / "artifacts.py",
        ROOT / "dashboard" / "i18n.py",
        ROOT / "dashboard" / "theme.py",
        *(ROOT / "dashboard" / "components").rglob("*.py"),
        *(ROOT / "dashboard" / "views").rglob("*.py"),
    ]
    for path in active_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(FORBIDDEN_PREFIXES) for alias in node.names), path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(FORBIDDEN_PREFIXES), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in FORBIDDEN_CALLS, f"{path}: forbidden call {node.func.attr}"
