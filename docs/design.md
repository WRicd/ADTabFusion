# AD-TabFusion Dashboard Design

This document records the implemented presentation contract derived from the root `design.md`. The root file remains the source design brief; this copy describes the public-facing dashboard delivered with the project.

## Product Intent

The dashboard is a read-only evidence viewer for frozen Phase A–D results. It supports scientific review and presentation without exposing participant-level data or changing model artifacts.

## Information Architecture

The persistent sidebar contains eight pages:

1. Executive Summary
2. Data & Cohort
3. Scientific Guardrails
4. Transition-Aware Model
5. MCI Progression Risk
6. Calibration & Uncertainty
7. External Replay
8. Reproducibility

Chinese and English are switched globally from the sidebar. The four presentation screenshots cover Executive Summary, Transition-Aware Model, MCI Progression Risk, and Calibration & Uncertainty.

## Visual System

- Calm biomedical palette: navy for hierarchy, blue and teal for primary evidence, amber for caution, and red only for critical limitations.
- White page surface with restrained gray panels and one-pixel borders.
- Cards use stable dimensions and a maximum 8 px visible corner radius.
- CN, MCI, and AD retain a fixed order and fixed class colors across charts.
- English metric names are preserved to match frozen artifact schemas and reports.
- Charts are built with Altair and aggregate values loaded from frozen CSV/JSON artifacts.

Implementation tokens live in `dashboard/theme.py`; layout styling lives in `dashboard/styles.css`; reusable components live in `dashboard/components/`.

## Evidence Language

- **Internal validation:** model development and ablation evidence.
- **Locked temporal test:** primary internal extrapolation evidence.
- **Exploratory post-hoc replay:** post-hoc D4 evidence only.

Every D4 view must display: **Exploratory post-hoc replay — not independent confirmatory validation**.

## Privacy and Safety

- Do not render RID, PTID, participant rows, raw predictions, or local absolute paths.
- Do not import training modules or call fitting, calibration, serialization, or artifact-writing routines.
- Missing optional artifacts render an empty state and do not crash navigation.
- Artifact paths shown in the UI are restricted to project-relative or basename-only forms.

## Test Contract

Dashboard tests under `tests/dashboard/` verify:

- theme tokens and fixed diagnosis ordering;
- registration of all eight pages;
- safe CSV/JSON loading and schema checks;
- absence of training imports and artifact writes;
- consistency with frozen Phase D metrics;
- required D4 limitation labels;
- graceful optional-artifact empty states.

