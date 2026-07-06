# Leakage Check

Task mode: `all_visits`

## Excluded Columns

- `DX`: Target diagnosis label; including it would directly leak the answer.
- `DXCHANGE`: Diagnosis transition code derived from diagnosis state; leaks target information.
- `DX_bl`: Baseline diagnosis is diagnosis-derived; excluded from main all-visits model.
- `VISCODE`: Visit code may encode follow-up timing and is excluded by default.

## Notes

`FAQ_bl` is treated as a baseline covariate, not as current-visit FAQ.
Subject-level splitting is required for every task mode.