# Overview Figures

Generated for 16:9 PPT presentation slides.

## Run Command

```bash
python scripts/generate_overview_figures.py
```

## Figure Guide

### `pipeline_overview.png`

- PPT Page 1: Project workflow
- Purpose: Shows the end-to-end local analysis pipeline.

### `dataset_modalities_overview.png`

- PPT Page 2: Dataset and modalities
- Purpose: Summarizes current D1_D2 scale, active modalities, and unavailable data.

### `model_performance_overview.png`

- PPT Page 3: Baseline model comparison
- Purpose: Compares Accuracy, Balanced Accuracy, Macro F1, and ROC-AUC OvR.

### `modality_ablation_overview.png`

- PPT Page 4: Modality ablation
- Purpose: Ranks modality combinations by Macro F1 or Balanced Accuracy.

### `missing_modality_overview.png`

- PPT Page 5: Missing-modality robustness
- Purpose: Shows performance drop after test-time modality masking.

### `explainability_overview.png`

- PPT Page 6: Explainability
- Purpose: Shows top feature importance and modality-level importance.
