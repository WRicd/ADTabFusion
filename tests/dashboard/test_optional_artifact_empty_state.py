from pathlib import Path

from dashboard.artifacts import load_csv
from dashboard.components.empty_state import empty_state


class StubContainer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def markdown(self, message: str, unsafe_allow_html: bool = False) -> None:
        self.calls.append((message, unsafe_allow_html))


def test_missing_optional_csv_can_render_empty_state(tmp_path: Path) -> None:
    frame, error = load_csv(tmp_path / "optional.csv")
    assert frame.empty
    assert error

    container = StubContainer()
    empty_state(error, container=container)
    assert len(container.calls) == 1
    assert "ad-empty" in container.calls[0][0]
    assert container.calls[0][1] is True
