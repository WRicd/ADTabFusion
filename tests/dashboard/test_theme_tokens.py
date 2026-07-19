from dashboard.theme import CLASS_COLORS, CLASS_ORDER, COLORS, RADIUS, SPACING


def test_required_theme_tokens_are_defined() -> None:
    assert {"navy", "blue", "teal", "green", "amber", "red", "ink", "muted", "border", "surface"} <= COLORS.keys()
    assert SPACING == {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}
    assert max(RADIUS.values()) <= 12


def test_diagnosis_order_and_colors_are_fixed() -> None:
    assert CLASS_ORDER == ["CN", "MCI", "AD"]
    assert list(CLASS_COLORS) == CLASS_ORDER

