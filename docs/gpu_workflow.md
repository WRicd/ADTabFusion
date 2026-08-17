# GPU Workflow and Model Extensions

This document covers the GPU-accelerated and deep tabular extensions added on
top of the frozen Phase A–D evidence.

## Scientific boundary

The frozen Phase A–D results remain the **scikit-learn baseline** and are the
numbers reported in the README and dashboard. Everything described here is an
**exploratory extension**:

- GPU/deep runs write to a separate `output_dir` (`outputs/phase_d_gpu`), so no
  frozen artifact is ever overwritten.
- Hyperparameter search reads **train and validation only**. The locked
  temporal test split is never loaded into a search loop; this is enforced by
  [`tests/test_tuning_never_sees_test_split.py`](../tests/test_tuning_never_sees_test_split.py).
- `tests/test_model_defaults_unchanged.py` pins the sklearn defaults so adding
  new models cannot silently shift previously frozen numbers.

## Hardware

Developed and verified on an **RTX 4060 Laptop GPU (8 GB)** with PyTorch
CUDA 12.6. Everything falls back to CPU automatically when CUDA is absent, so
CI and non-GPU machines run the same code paths.

## Available models

| Config name | Backend | GPU | Notes |
|---|---|---|---|
| `logistic_regression` | scikit-learn | – | Frozen baseline |
| `random_forest` | scikit-learn | – | Frozen baseline |
| `hist_gradient_boosting` | scikit-learn | – | Frozen baseline |
| `xgboost` | XGBoost | ✅ `device=cuda` | Early stopping on validation |
| `lightgbm` | LightGBM | ⚠️ needs OpenCL build | Early stopping on validation |
| `torch_mlp` | PyTorch | ✅ | Residual MLP, batch-norm + dropout |
| `ft_transformer` | PyTorch | ✅ | Feature-Tokenizer Transformer |
| `soft_voting` | – | inherits | Weighted average of member probabilities |

### Enabling GPU

`use_gpu` resolves per-model first, then falls back to the global default, so
you can turn GPU on globally and opt individual models out:

```yaml
models:
  use_gpu: true            # global default
  lightgbm:
    use_gpu: false         # this one stays on CPU
```

> **LightGBM note:** the PyPI wheel is CPU-only. `use_gpu: true` requires a
> source build with OpenCL. XGBoost's CUDA support works out of the box.

### Early stopping

Set `early_stopping: true` on `xgboost`, `lightgbm`, `torch_mlp`, or
`ft_transformer`. The validation split is used **only** to decide when to
stop — it is never merged into the training data.

```yaml
models:
  xgboost:
    early_stopping: true
    early_stopping_rounds: 30
```

## Deep tabular models

Both PyTorch estimators implement the scikit-learn interface (`fit`,
`predict`, `predict_proba`) and accept `sample_weight`, so they drop into the
existing `Pipeline` and honour Phase D's subject-balanced pair weighting.

They consume the **already-preprocessed** dense matrix from
`build_preprocessor` (imputed, scaled, one-hot encoded).

```yaml
models:
  torch_mlp:
    hidden_sizes: [256, 128, 64]
    dropout: 0.25
    learning_rate: 0.001
    batch_size: 256
    max_epochs: 300
    patience: 25
    class_weight: balanced
    early_stopping: true
    use_gpu: true

  ft_transformer:
    d_token: 64          # reduced automatically if not divisible by n_heads
    n_blocks: 3
    n_heads: 8
    learning_rate: 0.0001
    batch_size: 128
    max_epochs: 200
    patience: 20
    use_gpu: true
```

`class_weight: balanced` reproduces scikit-learn's balanced class weighting
inside the loss, which matters for the CN/MCI/AD imbalance.

## Hyperparameter search

```bash
# Score trials on the Phase D validation split
python scripts/tune_hyperparameters.py --models xgboost lightgbm --trials 60 --gpu

# Subject-grouped 5-fold CV over train+validation (no subject spans a fold)
python scripts/tune_hyperparameters.py \
    --models ft_transformer --mode groupkfold --folds 5 --trials 40 --gpu
```

Results are written to `outputs/<phase>/tuning/`:

- `best_params_<model>.json` — ready to paste into a config
- `tuning_summary.csv` — cross-model comparison

Options: `--metric {macro_f1,roc_auc_ovr,log_loss}`, `--timeout <seconds>`,
`--no-mlflow`. `groupkfold` mode enables Optuna's `MedianPruner`, which stops
unpromising trials after their first folds.

## Ensembles

```yaml
models:
  soft_voting:
    members: [xgboost, lightgbm, torch_mlp]
    weights: [1.0, 1.0, 1.0]     # optional; normalized internally
```

Members are fitted through the same dispatch (so each gets its own GPU/early
stopping settings) and their probabilities are averaged. A member whose
optional dependency is missing is skipped with a warning; an unknown model
name fails loudly, because that is a config typo rather than a missing package.

## Experiment tracking

Runs log to a local MLflow store at `outputs/mlruns`:

```bash
mlflow ui --backend-store-uri outputs/mlruns --port 5001
```

Disable per config with `project.tracking.enabled: false`, or per CLI run with
`--no-mlflow`.

## Explainability

SHAP runs automatically for tree-based best models when
`explainability.run_shap: true`, writing:

- `figures/shap_summary_best_model.png` — per-feature value/impact beeswarm
- `figures/shap_feature_importance_best_model.png` — mean |SHAP| bar chart
- `metrics/shap_importance_best_model.csv`

Non-tree models fall back to permutation importance.

## Two findings from adding these models

Widening the candidate pool surfaced two issues in the existing pipeline that
are worth reading before trusting a new run:

- [Model selection is fragile at the current margin](model_selection_fragility.md)
  — with 7 candidates, single-split argmax selected a model that lost on the
  locked test by 1.3 points, on a validation margin 50× smaller than fold-to-fold
  noise.
- [Reproducibility note](reproducibility_note.md) — the frozen 0.881 requires
  scikit-learn 1.9.0; other versions give 0.885.

Empirically, the tree models beat the deep ones on this dataset
(`hist_gradient_boosting` 0.824 vs `ft_transformer` 0.773 by subject-grouped CV).
That is the expected result for a few thousand rows of tabular clinical data and
is a useful negative result, not a configuration failure.

## Suggested sequence

```bash
# 1. Search hyperparameters (validation only)
python scripts/tune_hyperparameters.py --models xgboost lightgbm torch_mlp --trials 60 --gpu

# 2. Copy best params into configs/phase_d_transition_gpu.yaml

# 3. Single final fit, then one-shot temporal test evaluation
python scripts/train_transition_aware_model.py --config configs/phase_d_transition_gpu.yaml

# 4. Compare against the frozen baseline
mlflow ui --backend-store-uri outputs/mlruns --port 5001
```
