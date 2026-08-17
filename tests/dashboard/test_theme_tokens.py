from dashboard.theme import (
    CLASS_COLORS,
    CLASS_COLORS_DARK,
    CLASS_COLORS_LIGHT,
    CLASS_ORDER,
    COLORS,
    DARK,
    LIGHT,
    RADIUS,
    SEQUENTIAL_DARK,
    SEQUENTIAL_LIGHT,
    SPACING,
)


def test_required_theme_tokens_are_defined() -> None:
    assert {"navy", "blue", "teal", "green", "amber", "red", "ink", "muted", "border", "surface"} <= COLORS.keys()
    assert SPACING == {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}
    assert max(RADIUS.values()) <= 12


def test_diagnosis_order_and_colors_are_fixed() -> None:
    assert CLASS_ORDER == ["CN", "MCI", "AD"]
    assert list(CLASS_COLORS) == CLASS_ORDER
    assert list(CLASS_COLORS_LIGHT) == CLASS_ORDER
    assert list(CLASS_COLORS_DARK) == CLASS_ORDER


def test_validated_palette_hexes_are_pinned() -> None:
    """These exact values passed the palette validator in both modes.

    Editing a hex without re-running the validator can silently drop the
    palette below the colour-vision-deficiency or contrast floors, so the
    values are pinned here; dashboard/theme.py records the measurements.
    """
    assert CLASS_COLORS_LIGHT == {"CN": "#2a78d6", "MCI": "#eda100", "AD": "#e34948"}
    assert CLASS_COLORS_DARK == {"CN": "#3987e5", "MCI": "#c98500", "AD": "#e34948"}
    assert LIGHT["surface"] == "#ffffff"
    assert DARK["surface"] == "#1a1a19"


def test_sequential_ramps_are_single_hue_two_stop() -> None:
    """Magnitude uses one hue light->dark, never a rainbow."""
    for ramp in (SEQUENTIAL_LIGHT, SEQUENTIAL_DARK):
        assert len(ramp) == 2
        assert all(value.startswith("#") for value in ramp)


def test_light_and_dark_define_the_same_token_names() -> None:
    """A token defined in one mode but not the other renders as unset CSS."""
    assert LIGHT.keys() == DARK.keys()
