# Phase D Transition-Aware Progression Modeling

## Primary Result: Locked D1/D2 Temporal Test

All feature, model, calibration, and abstention decisions were made without D4. The primary claim uses the latest 15% of D1/D2 subjects by index date.

| Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC OvR | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Transition-aware | 0.884 | 0.878 | 0.881 | 0.958 | 0.373 | 0.190 |

## Transition Ablation

| Inputs | Model | Validation Macro F1 |
|---|---|---:|
| features_only | logistic_regression | 0.645 |
| features_plus_forecast | logistic_regression | 0.644 |
| features_plus_source_dx | hist_gradient_boosting | 0.810 |
| features_plus_source_dx_forecast | hist_gradient_boosting | 0.820 |

The current source diagnosis provides the largest gain; forecast_months adds a smaller additional improvement while preserving an explicit future-prediction formulation.

## MCI-to-AD Landmark Risk

| Horizon | Subjects | ROC-AUC | PR-AUC | Balanced Accuracy | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 12 months | 68 | 0.763 | 0.474 | 0.621 | 0.583 |
| 24 months | 53 | 0.797 | 0.699 | 0.726 | 0.723 |
| 36 months | 44 | 0.804 | 0.770 | 0.731 | 0.733 |
| 48 months | 22 | 0.694 | 0.919 | 0.681 | 0.581 |

Negative labels require sufficient follow-up; subjects with inadequate observation time are censored.

## Calibration

Validation selected `isotonic` calibration. On the locked temporal test, Log Loss changed from 0.373 to 0.352; Brier changed from 0.190 to 0.191.

## Selective Prediction

The validation-frozen abstention rule retained 81.1% of locked temporal-test pairs and achieved Macro F1 0.926 with error rate 0.073.

## Exploratory Post-Hoc D4 Replay

> D4 replay is not independent confirmatory validation because D4 was already accessed in Phase C.

Frozen replay Macro F1: 0.734; ROC-AUC OvR: 0.872. No model, feature, calibration, or threshold was changed.

## Reproducibility

- Phase C verification gate: PASS
- D3-compatible profiles use D1/D2 and D3 schemas only
- Subject temporal split is date-only and frozen
- Pair weights sum equally per RID
- Calibration and abstention threshold use validation subjects only
- Phase D frozen artifact registry is checked before and after D4 replay

Final success gate: **PASS**
