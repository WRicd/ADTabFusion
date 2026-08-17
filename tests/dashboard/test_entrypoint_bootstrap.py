"""Guardrail: `streamlit run dashboard/app.py` must work from any directory.

Streamlit puts the entry script's *own folder* on sys.path, not the project
root, so `import dashboard.*` raises ModuleNotFoundError unless app.py adds the
root itself. This regressed once and the failure only appears in the browser,
not in the server log, so it is easy to miss.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "dashboard" / "app.py"


def test_entrypoint_adds_the_project_root_to_sys_path() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))
    inserts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "insert"
    ]
    assert inserts, "app.py must bootstrap sys.path with the project root"


def test_dashboard_imports_resolve_under_streamlits_sys_path() -> None:
    """Reproduce Streamlit's sys.path exactly and confirm the imports resolve."""
    probe = textwrap.dedent(
        f"""
        import os, sys
        script = r"{APP}"
        # streamlit.web.bootstrap: sys.path.insert(0, dirname(main_script_path))
        sys.path = [os.path.dirname(script)] + [
            p for p in sys.path if not p.startswith(r"{ROOT}")
        ]
        try:
            import dashboard.i18n
            print("UNEXPECTED_OK")
        except ModuleNotFoundError:
            pass

        from pathlib import Path
        root = Path(script).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import dashboard.i18n, dashboard.theme, dashboard.artifacts, dashboard.charts
        print("BOOTSTRAP_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(ROOT.parent),  # deliberately not the project root
        timeout=180,
    )
    assert "BOOTSTRAP_OK" in result.stdout, result.stdout + result.stderr
