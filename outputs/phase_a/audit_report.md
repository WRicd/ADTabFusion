# Full TADPOLE Leakage Audit

Task: CN / MCI / AD internal diagnosis classification

Audited columns: 1907
Primary whitelist: 69
Blacklist: 1552

## Exclusion summary

| Rule | Columns |
|---|---:|
| exclude_administrative | 47 |
| exclude_duplicate | 330 |
| exclude_high_missing | 1115 |
| exclude_identifier | 13 |
| exclude_leakage | 15 |
| exclude_unresolved | 32 |

## Direct leakage and split-only fields

| Column | Modality | Reason |
|---|---|---|
| `RID` | identifier | Identifier is retained only for joins and subject-level splits. |
| `PTID` | identifier | Identifier is retained only for joins and subject-level splits. |
| `VISCODE` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `SITE` | administrative | Administrative or data-construction field. |
| `D1` | administrative | Administrative or data-construction field. |
| `D2` | administrative | Administrative or data-construction field. |
| `COLPROT` | administrative | Administrative or data-construction field. |
| `ORIGPROT` | administrative | Administrative or data-construction field. |
| `EXAMDATE` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `DX_bl` | baseline_diagnosis | Target-derived or temporal field is excluded from model inputs. |
| `DXCHANGE` | label | Target-derived or temporal field is excluded from model inputs. |
| `FSVERSION` | administrative | Administrative or data-construction field. |
| `DX` | label | Target-derived or temporal field is excluded from model inputs. |
| `EXAMDATE_bl` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `FSVERSION_bl` | administrative | Administrative or data-construction field. |
| `Years_bl` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `Month_bl` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `update_stamp` | administrative | Administrative or data-construction field. |
| `EXAMDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `VERSION_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `LONISID_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | identifier | Identifier is retained only for joins and subject-level splits. |
| `LONIUID_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | identifier | Identifier is retained only for joins and subject-level splits. |
| `IMAGEUID_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | identifier | Identifier is retained only for joins and subject-level splits. |
| `RUNDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `STATUS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `OVERALLQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `TEMPQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `FRONTQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `PARQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `INSULAQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `OCCQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `BGQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `CWMQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `VENTQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `update_stamp_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16` | administrative | Administrative or data-construction field. |
| `EXAMDATE_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `VERSION_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `LONISID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | identifier | Identifier is retained only for joins and subject-level splits. |
| `LONIUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | identifier | Identifier is retained only for joins and subject-level splits. |
| `IMAGEUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | identifier | Identifier is retained only for joins and subject-level splits. |
| `RUNDATE_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `STATUS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `OVERALLQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `TEMPQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `FRONTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `PARQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `INSULAQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `OCCQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `BGQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `CWMQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `VENTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `update_stamp_UCSFFSX_11_02_15_UCSFFSX51_08_01_16` | administrative | Administrative or data-construction field. |
| `EXAMDATE_BAIPETNMRC_09_12_16` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `VERSION_BAIPETNMRC_09_12_16` | administrative | Administrative or data-construction field. |
| `LONIUID_BAIPETNMRC_09_12_16` | identifier | Identifier is retained only for joins and subject-level splits. |
| `RUNDATE_BAIPETNMRC_09_12_16` | administrative | Administrative or data-construction field. |
| `STATUS_BAIPETNMRC_09_12_16` | administrative | Administrative or data-construction field. |
| `update_stamp_BAIPETNMRC_09_12_16` | administrative | Administrative or data-construction field. |
| `EXAMDATE_UCBERKELEYAV45_10_17_16` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `update_stamp_UCBERKELEYAV45_10_17_16` | administrative | Administrative or data-construction field. |
| `EXAMDATE_UCBERKELEYAV1451_10_17_16` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `update_stamp_UCBERKELEYAV1451_10_17_16` | administrative | Administrative or data-construction field. |
| `EXAMDATE_DTIROI_04_30_14` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `VERSION_DTIROI_04_30_14` | administrative | Administrative or data-construction field. |
| `LONIUID_1_DTIROI_04_30_14` | identifier | Identifier is retained only for joins and subject-level splits. |
| `LONIUID_2_DTIROI_04_30_14` | identifier | Identifier is retained only for joins and subject-level splits. |
| `LONIUID_3_DTIROI_04_30_14` | identifier | Identifier is retained only for joins and subject-level splits. |
| `LONIUID_4_DTIROI_04_30_14` | identifier | Identifier is retained only for joins and subject-level splits. |
| `RUNDATE_DTIROI_04_30_14` | administrative | Administrative or data-construction field. |
| `STATUS_DTIROI_04_30_14` | administrative | Administrative or data-construction field. |
| `update_stamp_DTIROI_04_30_14` | administrative | Administrative or data-construction field. |
| `EXAMDATE_UPENNBIOMK9_04_19_17` | visit_time | Target-derived or temporal field is excluded from model inputs. |
| `RUNDATE_UPENNBIOMK9_04_19_17` | administrative | Administrative or data-construction field. |
| `COMMENT_UPENNBIOMK9_04_19_17` | administrative | Administrative or data-construction field. |
| `update_stamp_UPENNBIOMK9_04_19_17` | administrative | Administrative or data-construction field. |

## Guardrails

- D4 is excluded from training, preprocessing fit, feature selection and tuning.
- D3 is an external transform-only input and is not used to fit the primary model.
- RID is retained only for subject-level splitting and D3/D4 matching.
- Visit and date fields are retained for audit or matching but excluded from diagnostic features.
- Duplicate source/date variants are reduced to one representative per measurement family.
