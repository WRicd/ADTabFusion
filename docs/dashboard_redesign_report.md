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
