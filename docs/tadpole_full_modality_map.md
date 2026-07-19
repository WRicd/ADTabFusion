# Full TADPOLE Modality Map

Assignments prioritize exact dictionary metadata, especially `TBLNAME` for DTI, PET and CSF sources.

| Modality | Total columns | Best missing rate | Eligible at 70% | Primary |
|---|---:|---:|---:|---:|
| administrative | 47 | 0.000 | 0 | 0 |
| baseline_diagnosis | 1 | 0.000 | 0 | 0 |
| cognitive | 20 | 0.000 | 10 | 10 |
| csf | 7 | 0.814 | 0 | 0 |
| demographic | 6 | 0.000 | 6 | 6 |
| genetic | 1 | 0.001 | 1 | 1 |
| identifier | 13 | 0.000 | 0 | 0 |
| label | 2 | 0.301 | 0 | 0 |
| mri_dti | 230 | 0.938 | 0 | 0 |
| mri_structural | 716 | 0.007 | 336 | 50 |
| pet_amyloid | 242 | 0.549 | 1 | 1 |
| pet_fdg | 336 | 0.283 | 1 | 1 |
| pet_other | 242 | 0.993 | 0 | 0 |
| unknown | 32 | 0.000 | 0 | 0 |
| visit_time | 12 | 0.000 | 0 | 0 |

## Source rules

- `DTIROI` dictionary rows map to `mri_dti`.
- `UCSFFSL` and `UCSFFSX` dictionary rows map to `mri_structural`.
- `BAIPETNMRC` maps to `pet_fdg`.
- `UCBERKELEYAV45` maps to `pet_amyloid`.
- `UCBERKELEYAV1451` maps to `pet_other`.
- `UPENNBIOMK9`, `ABETA_*`, `TAU_*` and `PTAU_*` map to `csf`.

> DTI and CSF are present but do not pass the default 70% missing-rate threshold in the full-row primary cohort. They require modality-specific cohort experiments rather than silent inclusion in the primary model.
