# AD-TabFusion Project Brief

## One Sentence

AD-TabFusion uses longitudinal multimodal clinical tables to predict future CN/MCI/AD diagnosis and MCI-to-AD progression while making temporal validation, calibration, and uncertainty explicit.

## Motivation

Alzheimer’s disease research cohorts combine demographic, cognitive, MRI-derived, and genetic variables across irregular repeat visits. Random row-level evaluation can overstate generalization when the same subject appears across splits or future information leaks into predictors. This project therefore emphasizes subject isolation, temporal extrapolation, frozen evaluation, and auditable evidence labels.

## Data and Tasks

- Primary active modalities: demographic, cognitive/clinical, MRI-derived, and APOE4.
- Sparse modalities: PET, CSF, and DTI; these support restricted-cohort analyses and are not claimed as complete primary modalities.
- Transition task: predict future CN/MCI/AD at a 6–60 month forecast horizon.
- MCI risk task: estimate progression to AD within 12, 24, 36, and 48 months.

## Key Results

| Result | Evidence split | Value |
|---|---|---:|
| Transition Macro F1 | Locked temporal test | 0.881 |
| Transition ROC-AUC OvR | Locked temporal test | 0.958 |
| 24-month MCI ROC-AUC | Locked temporal test, n=53 | 0.797 |
| 36-month MCI ROC-AUC | Locked temporal test, n=44 | 0.804 |
| Isotonic Log Loss | Locked temporal test | 0.352 |
| 80% target coverage / Macro F1 | Locked temporal test | 81.1% / 0.926 |

## Scientific Boundary

Models, calibration, and selective prediction thresholds are selected on validation data only; the locked temporal test is not used for refitting. Every D4 result is an **exploratory post-hoc replay**, not independent confirmatory validation. The software is retrospective research infrastructure, not a clinical decision system.

## Presentation

The bilingual Streamlit dashboard reads frozen CSV/JSON files under `outputs` only. Public pages and screenshots contain aggregate results and never display RID, PTID, or participant-level rows.

