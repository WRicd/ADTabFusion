# Frozen Result Cards

These short cards are presentation-ready summaries. Values are rounded for display and trace back to frozen Phase D CSV artifacts and `outputs/reports/phase_d_final_report.md`.

## Card 1 — Temporal Transition Model

**Claim:** Future CN/MCI/AD state remains predictable under a locked temporal test.

**Evidence:** Accuracy 0.884 · Balanced Accuracy 0.878 · Macro F1 0.881 · ROC-AUC OvR 0.958.

**Source:** `outputs/phase_d/temporal_validation/transition_model_results.csv`

## Card 2 — Transition Inputs Matter

**Claim:** Current diagnosis provides the main ablation gain; forecast horizon adds a smaller additional gain.

**Evidence:** Validation Macro F1 0.645 for features only, 0.810 with source diagnosis, and 0.820 with source diagnosis plus forecast horizon.

**Source:** `outputs/phase_d/internal_validation/transition_ablation.csv`

## Card 3 — MCI Progression

**Claim:** The 24–36 month windows show the most stable discrimination in the locked temporal test.

**Evidence:** 24-month ROC-AUC 0.797 (n=53); 36-month ROC-AUC 0.804 (n=44).

**Caution:** The 48-month estimate uses n=22 and is exploratory.

**Source:** `outputs/phase_d/temporal_validation/mci_risk_metrics.csv`

## Card 4 — Calibration

**Claim:** Validation-selected isotonic calibration improves probabilistic fit but not every calibration metric.

**Evidence:** Log Loss 0.373 → 0.352; ECE 0.047 → 0.039; Brier Score 0.190 → 0.191.

**Source:** `outputs/phase_d/calibration/calibration_results.csv`

## Card 5 — Selective Prediction

**Claim:** A validation-frozen confidence threshold identifies a higher-performing retained subset.

**Evidence:** 80% target coverage produces 81.1% observed coverage, Macro F1 0.926, and 7.3% error on the locked temporal test.

**Source:** `outputs/phase_d/uncertainty/selective_prediction.csv`

## Card 6 — D4 Replay

**Label:** **Exploratory post-hoc replay — not independent confirmatory validation.**

**Evidence:** Full replay Macro F1 0.734 and ROC-AUC OvR 0.872; frozen selective replay coverage 45.7% and Macro F1 0.853.

**Source:** `outputs/phase_d/d4_replay/d4_replay_metrics.csv`
