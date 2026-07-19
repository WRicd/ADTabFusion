# Full TADPOLE Feature Catalog

Audited columns: 1907
Dictionary matches: 1888
Primary whitelist features: 69

## Recommendation summary

| Status | Columns |
|---|---:|
| exclude_administrative | 47 |
| exclude_duplicate | 330 |
| exclude_high_missing | 1115 |
| exclude_identifier | 13 |
| exclude_leakage | 15 |
| exclude_unresolved | 32 |
| include_optional | 286 |
| include_primary | 69 |

## Modality summary

| Modality | Columns |
|---|---:|
| mri_structural | 716 |
| pet_fdg | 336 |
| pet_other | 242 |
| pet_amyloid | 242 |
| mri_dti | 230 |
| administrative | 47 |
| unknown | 32 |
| cognitive | 20 |
| identifier | 13 |
| visit_time | 12 |
| csf | 7 |
| demographic | 6 |
| label | 2 |
| genetic | 1 |
| baseline_diagnosis | 1 |

## Primary whitelist

| Column | Modality | Source | Missing rate | D3 available |
|---|---|---|---:|---|
| `AGE` | demographic | ADNIMERGE | 0.000 | yes |
| `PTGENDER` | demographic | ADNIMERGE | 0.000 | yes |
| `PTEDUCAT` | demographic | ADNIMERGE | 0.000 | yes |
| `PTETHCAT` | demographic | ADNIMERGE | 0.000 | yes |
| `PTRACCAT` | demographic | ADNIMERGE | 0.000 | yes |
| `PTMARRY` | demographic | ADNIMERGE | 0.000 | yes |
| `APOE4` | genetic | ADNIMERGE | 0.001 | no |
| `ADAS13` | cognitive | ADNIMERGE | 0.307 | yes |
| `MMSE` | cognitive | ADNIMERGE | 0.299 | yes |
| `ICV` | mri_structural | ADNIMERGE | 0.376 | yes |
| `CDRSB_bl` | cognitive | ADNIMERGE | 0.000 | no |
| `ADAS11_bl` | cognitive | ADNIMERGE | 0.001 | no |
| `RAVLT_immediate_bl` | cognitive | ADNIMERGE | 0.003 | no |
| `RAVLT_learning_bl` | cognitive | - | 0.003 | no |
| `RAVLT_forgetting_bl` | cognitive | - | 0.003 | no |
| `RAVLT_perc_forgetting_bl` | cognitive | - | 0.004 | no |
| `FAQ_bl` | cognitive | ADNIMERGE | 0.004 | no |
| `MOCA_bl` | cognitive | ADNIMERGE | 0.550 | no |
| `FDG_bl` | pet_fdg | ADNIMERGE | 0.283 | no |
| `AV45_bl` | pet_amyloid | ADNIMERGE | 0.549 | no |
| `ST101SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST102CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST102SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST102TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST102TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST103CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST103SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST103TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST103TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST104CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST104SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST104TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST104TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST105CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST105SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST105TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST105TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST106CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST106SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST106TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST106TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST107CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST107SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST107TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST107TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST108CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST108SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST108TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST108TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST109CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST109SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST109TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST109TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST10CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST110CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST110SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST110TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST110TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST111CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST111SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST111TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST111TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST112SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST113CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST113SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST113TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST113TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST114CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |
| `ST114SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | mri_structural | UCSFFSX | 0.376 | yes |

> The CSV output contains the complete per-column audit. This Markdown report contains aggregate results and the primary whitelist only.
