# ADNI File Inventory

Scanned directory: `data\tadpole_challenge`
CSV files found: 4
Readable CSV files: 4
Unreadable files: 0

## Available modalities

| Modality | Available | Files | Key columns |
|---|---|---|---|
| data_dictionary | yes | TADPOLE_D1_D2_Dict.csv | FLDNAME, TBLNAME, TEXT, TYPE, UNITS, CODE |
| diagnosis | yes | TADPOLE_D1_D2.csv, TADPOLE_D3.csv, TADPOLE_D4_corr.csv | COLPROT, DX_bl, DXCHANGE, DX, Diagnosis |
| demographic | yes | TADPOLE_D1_D2.csv, TADPOLE_D3.csv, TADPOLE_D4_corr.csv | AGE, PTGENDER, PTEDUCAT, PTETHCAT, PTRACCAT, PTMARRY |
| apoe | yes | TADPOLE_D1_D2.csv | APOE4 |
| cognitive | yes | TADPOLE_D1_D2.csv, TADPOLE_D3.csv, TADPOLE_D4_corr.csv | CDRSB, ADAS11, ADAS13, MMSE, RAVLT_immediate, RAVLT_learning, RAVLT_forgetting, RAVLT_perc_forgetting, +10 more |
| mri_measurement | yes | TADPOLE_D1_D2.csv, TADPOLE_D3.csv, TADPOLE_D4_corr.csv | Ventricles, Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp, ICV, LB4, +4 more |
| pet_measurement | yes | TADPOLE_D1_D2.csv | FDG, PIB, AV45, FDG_bl, PIB_bl, AV45_bl, MCSUVR_BAIPETNMRC_09_12_16, MCSUVRWM_BAIPETNMRC_09_12_16, +251 more |
| biofluid_csf | yes | TADPOLE_D1_D2.csv | EXAMDATE_UPENNBIOMK9_04_19_17, PHASE_UPENNBIOMK9_04_19_17, BATCH_UPENNBIOMK9_04_19_17, KIT_UPENNBIOMK9_04_19_17, STDS_UPENNBIOMK9_04_19_17, RUNDATE_UPENNBIOMK9_04_19_17, ABETA_UPENNBIOMK9_04_19_17, TAU_UPENNBIOMK9_04_19_17, +3 more |

## File-level inventory

| File | Category | Rows | Columns | ID cols | Time cols | Key matched columns | Recommended use |
|---|---|---:|---:|---|---|---|---|
| data/tadpole_challenge/TADPOLE_D1_D2.csv | merged_adnimerge | 12741 | 1907 | RID, PTID | VISCODE, COLPROT, EXAMDATE | AGE, PTGENDER, PTEDUCAT, PTETHCAT, PTRACCAT, PTMARRY, CDRSB, ADAS11, +294 more | primary_multimodal_table |
| data/tadpole_challenge/TADPOLE_D1_D2_Dict.csv | data_dictionary | 1918 | 33 | - | - | FLDNAME, TBLNAME, TEXT, TYPE, UNITS, CODE | data_dictionary_reference |
| data/tadpole_challenge/TADPOLE_D3.csv | merged_adnimerge | 896 | 383 | RID | VISCODE, EXAMDATE, COLPROT | AGE, PTGENDER, PTEDUCAT, PTETHCAT, PTRACCAT, PTMARRY, ADAS13, MMSE, +7 more | primary_multimodal_table |
| data/tadpole_challenge/TADPOLE_D4_corr.csv | mri_measurement | 234 | 13 | RID | ScanDate | AGE, PTGENDER, ADAS13, MMSE, Ventricles, LB4, Years_bl | mri_measurement_table |

## Missing recommended categories

- PET measurements: found
- Biofluid / CSF measurements: found
- MRI measurements: found
- Cognitive assessments: found

## Next recommended downloads

- No missing second-batch modality category was detected.

> This inventory contains structural metadata only. No row-level source data is included.

## Dictionary-backed D1/D2 modality classification

| Modality | Available | Columns |
|---|---|---:|
| administrative | yes | 47 |
| baseline_diagnosis | yes | 1 |
| cognitive | yes | 20 |
| csf | yes | 7 |
| demographic | yes | 6 |
| genetic | yes | 1 |
| identifier | yes | 13 |
| label | yes | 2 |
| mri_dti | yes | 230 |
| mri_structural | yes | 716 |
| pet_amyloid | yes | 242 |
| pet_fdg | yes | 336 |
| pet_other | yes | 242 |
| unknown | yes | 32 |
| visit_time | yes | 12 |
