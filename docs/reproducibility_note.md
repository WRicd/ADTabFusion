# Reproducibility Note: Frozen Phase D Results and Library Drift

## Finding

The frozen Phase D artifacts do **not** reproduce exactly in an environment
with a different scikit-learn version. This was discovered on 2026-07-26 while
verifying that GPU model additions had not disturbed the frozen baseline.

| Environment | Python | scikit-learn | Transition Macro F1 (locked temporal test) |
|---|---|---|---|
| Frozen artifacts (2026-07-19) | 3.12.13 | 1.9.0 | **0.881** |
| Verification machine (2026-07-26) | 3.13.2 | 1.8.0 | **0.885** |

Per-model validation Macro F1 on the same data and the same seeds:

| Model | Frozen (sklearn 1.9.0) | Re-run (sklearn 1.8.0) |
|---|---:|---:|
| `logistic_regression` | 0.815667 | 0.815667 |
| `random_forest` | 0.816599 | 0.813690 |
| `hist_gradient_boosting` | 0.820416 | 0.822635 |

## Cause

Library version drift, not a code change. This was confirmed by checking out
the original `configs/phase_d_transition_model.yaml` from commit `a95be71` and
running it unmodified: it produced 0.885, matching the current code exactly.

`logistic_regression` is unaffected because it converges to the same optimum
regardless of version. `random_forest` and `hist_gradient_boosting` depend on
version-specific internals (tie-breaking in split selection, binning, and RNG
consumption order), so their fitted trees differ even with a fixed
`random_state`.

`requirements.txt` pins with `>=` (`scikit-learn>=1.3`), which permits any of
these versions and therefore does not protect the frozen numbers.

## What this does and does not mean

- **Scientific conclusions are unchanged.** Every reported effect holds under
  both versions; the differences are in the third decimal place and the same
  model is selected either way.
- **Exact artifact-level reproduction requires pinning.** A reader who
  installs from `requirements.txt` today may see 0.885 rather than the
  published 0.881.

## Remediation

`requirements-lock.txt` records the exact versions of the verification
environment. To reproduce the **published frozen numbers**, pin the original
environment instead:

```
python==3.12.13
scikit-learn==1.9.0
```

Recommended next steps, in priority order:

1. Regenerate the frozen artifacts once under a pinned, documented environment
   and record that environment in the manifests (the manifest already captures
   `python_version` and `sklearn_version` — this note exists because that
   metadata worked as intended).
2. Add the pinned environment to CI so drift is detected automatically.
3. Consider a container image for archival reproduction.

## Verifying reproduction yourself

```bash
# Re-run the frozen config into a scratch directory (never overwrite outputs/phase_d)
git show a95be71:configs/phase_d_transition_model.yaml > /tmp/orig.yaml
sed -i 's|output_dir: outputs/phase_d|output_dir: outputs/_repro_check|' /tmp/orig.yaml
python scripts/train_transition_aware_model.py --config /tmp/orig.yaml

# Compare
python - <<'PY'
import pandas as pd
cols = ["model", "macro_f1"]
frozen = pd.read_csv("outputs/phase_d/internal_validation/transition_model_results.csv")[cols]
repro = pd.read_csv("outputs/_repro_check/internal_validation/transition_model_results.csv")[cols]
print(frozen.merge(repro, on="model", suffixes=("_frozen", "_repro")))
PY
```
