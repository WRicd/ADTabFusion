# D3 Schema Compatibility Audit

> This audit was completed without reading D4 labels.

- Frozen whitelist features: 69
- Present in D3: 58 (84.1%)
- Absent in D3: 11
- Severe dtype mismatches: 0
- Primary deployment: `compact_random_forest`
- Sensitivity deployment: `full_hist_gradient_boosting`

## Decision

Full-primary compatibility requires at least 95% feature presence, every required modality, and no severe dtype mismatch.
The deployment order above is frozen before any D4 evaluation.

## Absent Features

- `APOE4`
- `CDRSB_bl`
- `ADAS11_bl`
- `RAVLT_immediate_bl`
- `RAVLT_learning_bl`
- `RAVLT_forgetting_bl`
- `RAVLT_perc_forgetting_bl`
- `FAQ_bl`
- `MOCA_bl`
- `FDG_bl`
- `AV45_bl`

## Modality Coverage

| Modality | Present | Total | Usable |
|---|---:|---:|---|
| cognitive | 2 | 10 | yes |
| demographic | 6 | 6 | yes |
| genetic | 0 | 1 | no |
| mri_structural | 50 | 50 | yes |
| pet_amyloid | 0 | 1 | no |
| pet_fdg | 0 | 1 | no |

## Unseen Categorical Levels

None detected.

## Frozen Alignment Policy

Absent columns are inserted in frozen training order with all values set to NaN. The fitted training imputer handles them; no D3 transformer is fitted or updated.
