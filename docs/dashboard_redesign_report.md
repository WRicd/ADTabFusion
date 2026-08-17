# Dashboard Redesign Report

## Scope

The Streamlit presentation layer was rebuilt from the root `design.md` without changing Phase A–D models, configurations, data splits, thresholds, calibrators, or frozen artifacts. The new dashboard is a read-only consumer of existing CSV/JSON files under `outputs`.

## Modified and Added Files

### Dashboard shell and styling

- `dashboard/app.py` — explicit bilingual eight-page navigation.
- `dashboard/theme.py` — shared color, spacing, radius, and diagnosis-class tokens.
- `dashboard/styles.css` — responsive biomedical presentation styling with fixed light chart surfaces.
- `dashboard/i18n.py` — global Chinese/English state.
- `dashboard/artifacts.py` — cached, schema-aware, read-only CSV/JSON loaders and safe path display.
- `dashboard/charts.py` — consistent white-background Altair rendering.

### Reusable components

- `dashboard/components/metric_card.py`
- `dashboard/components/section_header.py`
- `dashboard/components/status_badge.py`
- `dashboard/components/limitation_banner.py`
- `dashboard/components/chart_container.py`
- `dashboard/components/artifact_table.py`
- `dashboard/components/interpretation_box.py`
- `dashboard/components/empty_state.py`

### Registered views

- `dashboard/views/executive_summary.py`
- `dashboard/views/data_cohort.py`
- `dashboard/views/scientific_guardrails.py`
- `dashboard/views/transition_aware.py`
- `dashboard/views/mci_progression.py`
- `dashboard/views/calibration_uncertainty.py`
- `dashboard/views/external_replay.py`
- `dashboard/views/reproducibility.py`

The legacy `dashboard/pages/` files remain untouched but are not registered by the new `st.navigation` entry point.

### Public packaging

- `README.md` — compressed one-minute project summary.
- `docs/project_brief_zh.md`
- `docs/project_brief_en.md`
- `docs/demo_script_zh.md`
- `docs/result_cards.md`
- `docs/design.md`
- `docs/dashboard_redesign_report.md`

### Dashboard tests

- `tests/dashboard/test_theme_tokens.py`
- `tests/dashboard/test_required_pages.py`
- `tests/dashboard/test_artifact_loaders.py`
- `tests/dashboard/test_no_training_imports.py`
- `tests/dashboard/test_metric_consistency.py`
- `tests/dashboard/test_limitation_labels.py`
- `tests/dashboard/test_optional_artifact_empty_state.py`

## Verification

Command:

```bash
python -m pytest tests/dashboard tests/test_dashboard_utils.py -q
```

Result: **14 passed in 6.29 seconds**.

The tests verify the theme contract, eight required pages, read-only artifact loading, absence of training imports and artifact writes, consistency with frozen Phase D metrics, mandatory D4 limitation language, and optional-artifact empty states.

Browser QA used a 16:9 viewport and confirmed:

- the four priority pages render their metric cards and charts;
- Chinese navigation is complete and the language selector switches successfully to English;
- the D4 page displays `Exploratory post-hoc replay — not independent confirmatory validation` in two visible locations;
- no exact `RID` or `PTID` text appears on the captured priority pages or D4 replay page;
- chart canvases are nonblank and use a fixed light background.

## Public Screenshots

All screenshots are 1280×720 PNG files and contain aggregate evidence only.

- `docs/assets/dashboard_executive.png`
- `docs/assets/dashboard_transition.png`
- `docs/assets/dashboard_mci_risk.png`
- `docs/assets/dashboard_uncertainty.png`

## Remaining Limitations

- The 48-month MCI locked temporal test contains 22 subjects and remains small-sample exploratory evidence.
- PET, CSF, and DTI are sparse and cannot be presented as complete primary-model modalities.
- D4 is an exploratory post-hoc replay and is not independent confirmatory external validation.
- The dashboard depends on the current frozen artifact schemas; missing optional files render an empty state, while missing core evidence prevents the associated chart from appearing.
- This is retrospective research software and is not intended for individual clinical decision-making.
- The local `tests/.tmp-dashboard` cache may remain because the managed environment rejected recursive cleanup; it is untracked test output and does not affect project artifacts.

---

# Accessibility and Theme Pass (2026-07-26)

A follow-up pass to the redesign above. No model, config, split, threshold,
calibrator, or frozen artifact changed; the dashboard remains read-only.

## Palette is now validated, not chosen by eye

The previous diagnosis colours were replaced after running them through a
colour-vision/contrast validator. The original CN blue `#2F6B9A` failed the
chroma floor (0.097 — it reads gray). The replacements clear every hard gate:

| | CN | MCI | AD | Surface |
|---|---|---|---|---|
| Light | `#2a78d6` | `#eda100` | `#e34948` | `#ffffff` |
| Dark | `#3987e5` | `#c98500` | `#e34948` | `#1a1a19` |

Each mode retains exactly one advisory, and both are discharged the same way:

- **Light:** MCI amber sits at 2.11:1 against the surface (below 3:1).
- **Dark:** MCI/AD separate by CVD ΔE 6.2, inside the 6–8 band that is only
  permitted alongside a secondary encoding.

Darkening the amber to clear 3:1 was tested and rejected — it collapses the
MCI/AD CVD separation from ΔE 15.3 to 3.7, trading a relievable advisory for a
hard failure. The relief is therefore **direct value labels on every mark plus
a table view under every chart**, which `tests/dashboard/test_chart_accessibility.py`
enforces. The hexes are pinned in `tests/dashboard/test_theme_tokens.py` so a
future edit cannot silently drop below the floors without re-validating.

Magnitude encodings (confusion counts, cohort heatmaps) now use a single-hue
sequential ramp rather than an ad-hoc two-colour scale, and `series_color()`
raises rather than cycling hues past its defined slots.

## Opt-in dark mode

The original design pinned a light chart surface. That is preserved as the
**default** so published screenshots stay reproducible, but the surface is no
longer hardcoded: an *Appearance* control in the sidebar switches modes, and
every rule is written against custom properties that `apply_theme()` supplies
per mode. Dark is a selected set of steps validated against the dark surface,
not an automatic inversion of the light values. Callout fills are mixed against
the active surface via `color-mix`, with solid light-mode fallbacks for
browsers that lack it.

## Fixes

- **Latent `KeyError`** in the executive summary: `n_subjects` and `n_rows`
  were rendered but not declared as required columns, so an artifact missing
  them passed validation and then crashed mid-render instead of showing the
  empty state.
- **Deprecated API:** `use_container_width` passed its removal date
  (2025-12-31); all 7 call sites moved to `width=`, and the Streamlit floor
  moved to `>=1.49` accordingly.
- **Legacy code removed:** the 13 unregistered `dashboard/pages/` modules and
  `dashboard/dashboard_utils.py`. They were unreachable from `st.navigation`,
  duplicated the language selector, and — because the no-training-imports
  guardrail only covered `views/` and `components/` — were the one part of the
  dashboard that could have imported training code unchecked.

## Verification

`tests/dashboard/test_views_render.py` executes all eight views in both colour
modes against the frozen artifacts (16 render paths) plus the entry point.
Full suite: 170 passed.

---

# Theming Correction (2026-07-26, later same day)

The Appearance pass above shipped a sidebar radio that swapped custom CSS
tokens. Two bugs were reported against it: some tables kept their colours when
switching modes, and some icons failed to load. Both were root-caused from the
compiled Streamlit 1.57 bundle; both are now fixed at the mechanism.

## Bug 1 — tables did not follow the switch

`st.dataframe` is glide-data-grid. Its cells are painted with
`ctx.fillStyle` / `fillRect` onto a `<canvas>`, using Streamlit's internal
theme object (`bgCell ← backgroundColor`, `textDark ← textColor`,
`bgHeader ← dataframeHeaderBackgroundColor`, `borderColor ← dataframeBorderColor`).

**No CSS can repaint a bitmap.** Streamlit 1.57 also exposes *zero* theme CSS
custom properties, so there was nothing for a stylesheet to hook. The sidebar
radio moved only `--ad-*`, which reaches the wrapper border and nothing else.
The tables that *did* update were the HTML ones; the canvas ones could not.

Theme options are additionally non-scriptable — `st.set_option("theme.primaryColor", …)`
raises *"cannot be set on the fly"* — so **no Python control of any kind can
repaint that canvas**. The radio was structurally incapable of the job.

**Fix:** `.streamlit/config.toml` now defines `[theme.light]` and
`[theme.dark]`, which is the only server-side surface that reaches the grid.
Verified by building the protobuf the frontend actually receives:

```
LIGHT bg / text / dfHeader: #ffffff / #17212b / #f6f8fa
DARK  bg / text / dfHeader: #1a1a19 / #f2f3f4 / #232322
both modes registered -> True
```

Defining **both** sections is mandatory, not stylistic: with only a flat
`[theme]` block the frontend registers a single theme and its switcher hides
itself.

## Bug 2 — icons failed to load

Material icons are **font ligatures**: Streamlit renders
`<span data-testid="stIconMaterial">keyboard_arrow_down</span>` and relies on
the *Material Symbols Rounded* font to turn that text into a glyph.

`styles.css` contained:

```css
html, body, [class*="st-"] { font-family: Inter, "Segoe UI", Arial, sans-serif; }
```

Streamlit's emotion cache key is literally `st-emotion-cache` and its styletron
prefix is `st-`, so `[class*="st-"]` matched **every** Streamlit element,
including those icon spans. Specificity tied at (0,1,0) and the injected
stylesheet won on document order (`st.markdown` injects into `<body>`, emotion
into `<head>`). The ligature therefore failed and the icon typeset as its own
name. The font was never missing — it ships locally as
`MaterialSymbols-Rounded.VqUtTjSV.woff2` — it was overridden.

This also explains why only *some* icons broke: SVG-based icons (dataframe
toolbar, select chevrons) and inline-styled `:material/…` markdown icons were
unaffected.

**Fix:** the rule is deleted. Typography now comes from `[theme] font` in
config.toml, which sets the body font without touching the icon font. An
explicit `[data-testid="stIconMaterial"]` guard restores the icon font should
anything override it again, and a test fails if a `[class*="st-"]` font rule
is ever reintroduced.

## Architecture: one switch, not two

`[data-testid="stHeader"] { display: none; }` had hidden the toolbar — which is
where Streamlit's **only** runtime theme switch (System / Light / Dark) lives.
Hiding it is what made a hand-rolled radio seem necessary in the first place.

The header is now visible but transparent, `client.toolbarMode = "viewer"`
hides the developer affordances, and the sidebar radio is **removed** in favour
of a caption pointing at the native control. `theme.get_mode()` now *reads*
Streamlit's active theme instead of maintaining a competing one, so the canvas
and the `.ad-*` layer can no longer disagree.

### Deliberate trade-offs

- **The sidebar radio is gone.** This is a documented UX change. It is
  unavoidable: the native switch is the only one that can repaint the canvas,
  and keeping both would reproduce the reported desync.
- **The native default is "System"** (OS-following), so *light is the default*
  now holds as a screenshot procedure — pick Light explicitly before capturing —
  rather than as a server setting. `base = "light"` is set, and the session
  override keeps tests and headless capture deterministic.
- **One-rerun lag.** Streamlit sends no rerun when the theme changes, so
  `st.context.theme.type` is briefly stale (upstream issue #11920). The
  stylesheet mirrors the dark tokens under `prefers-color-scheme` so the custom
  layer keeps up in the common System case; Altair colours may lag a single
  interaction after an explicit flip.
- The hand-written BaseWeb `!important` overrides were deleted — they existed
  only because Streamlit's theme was never set, and they left a white popover
  frame around darkened rows.

The validated diagnosis palette is unchanged. Cross-mode reuse was tested and
rejected: the light palette fails the dark lightness band, and reusing the dark
palette in light mode would drop MCI/AD colourblind separation from ΔE 15.3 to
6.2. Mode-specific hues stay.

## Verification

`tests/dashboard/test_theme_config.py` (8 tests) asserts config/Python/CSS stay
in sync, the header is never hidden again, and no `[class*="st-"]` font rule
returns. Full suite: **180 passed**.
