# Model Selection Is Fragile at the Current Margin

## Finding

The Phase D selection rule picks the model with the highest Macro F1 on a
single validation split. When the candidate pool was widened from 3 models to 7
(adding CUDA XGBoost, LightGBM, a PyTorch MLP, and an FT-Transformer), the rule
selected a model that **lost** on the locked temporal test.

| Model | Validation (single split) | Subject-grouped 5-fold CV | Locked temporal test |
|---|---:|---:|---:|
| `hist_gradient_boosting` | 0.822635 | **0.8243 ± 0.0065** | **0.885** |
| `ft_transformer` | **0.822760** | 0.7728 ± 0.0191 | 0.872 |

The FT-Transformer won selection by **0.000126** Macro F1 — and then scored
1.3 points lower on the test.

## Why the margin is meaningless

The selection margin is roughly **50× smaller** than the fold-to-fold standard
deviation of the same metric on the same data (0.000126 vs 0.0065). It carries
no information. Subject-grouped cross-validation — a far more stable estimator
than one split — puts the two models 0.05 apart, about 8 standard deviations,
in the *opposite* direction from what the single split indicated.

The single validation split therefore did not merely fail to separate the
models; it ranked them backwards.

## Why this did not surface before

With three closely-related sklearn models the top-2 validation gap was large
enough relative to noise that argmax was stable. The failure appears only once
the pool contains models with genuinely different variance profiles: the
FT-Transformer's fold-to-fold spread (0.0191) is 3× the tree model's (0.0065),
so it wins a single-split lottery more often than it deserves.

## Recommendation

Do not decide this silently in code — it changes a published selection rule.
Two defensible options:

**Option A — tolerance band (recommended).** Keep validation Macro F1 as the
primary criterion, but treat differences below a preregistered tolerance as
ties and fall through to the existing complexity tie-break:

```python
best = full_results["macro_f1"].max()
tolerance = 0.005  # preregister this value
contenders = full_results[full_results["macro_f1"] >= best - tolerance]
selected_row = contenders.sort_values(["complexity", "log_loss", "brier_score"], ascending=[True, True, True]).iloc[0]
```

With a 0.005 band, both models above are contenders and the simpler
`hist_gradient_boosting` wins — which is also the model that generalizes better
here.

**Option B — select on cross-validation.** Replace the single-split criterion
with the mean of subject-grouped K-fold CV over train+validation. This is
statistically stronger but more expensive, and it changes what "validation
Macro F1" means in every existing report and dashboard page.

Either way the tolerance or fold count must be fixed **before** looking at the
test split, or the guarantee that the test is a one-shot measurement is lost.

## Caveat on the numbers above

These runs used scikit-learn 1.8.0, not the 1.9.0 that produced the frozen
artifacts, so the tree-model figures differ slightly from the published ones.
See [reproducibility note](reproducibility_note.md). The selection-margin
argument does not depend on that difference: it compares a margin against a
variance measured in the same environment.
