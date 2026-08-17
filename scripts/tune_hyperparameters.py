"""Optuna hyperparameter search for the Phase D transition-aware task.

The locked temporal test split is never loaded into the search. Trials are
scored either on the Phase D validation split (``--mode holdout``) or with
subject-grouped K-fold CV over train+validation (``--mode groupkfold``).

Examples
--------
    python scripts/tune_hyperparameters.py --models xgboost lightgbm --trials 60 --gpu
    python scripts/tune_hyperparameters.py --models ft_transformer --mode groupkfold --gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.feature_groups import infer_feature_types
from src.phase_d.transition_model import ABLATIONS, load_transition_data
from src.preprocessing import build_preprocessor
from src.tuning import tune_many

LOGGER = logging.getLogger("tune")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/phase_d_transition_model.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lightgbm"],
        help="Models to tune (xgboost, lightgbm, hist_gradient_boosting, random_forest, torch_mlp, ft_transformer).",
    )
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-model wall-clock limit in seconds.")
    parser.add_argument("--metric", default="macro_f1", choices=["macro_f1", "roc_auc_ovr", "log_loss"])
    parser.add_argument("--mode", default="holdout", choices=["holdout", "groupkfold"])
    parser.add_argument("--folds", type=int, default=5, help="Folds when --mode groupkfold.")
    parser.add_argument("--gpu", action="store_true", help="Enable CUDA for models that support it.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    output_dir = Path(config["project"]["output_dir"])

    LOGGER.info("Loading Phase D transition pairs ...")
    pairs, base_features, _ = load_transition_data(config)
    features = ABLATIONS["features_plus_source_dx_forecast"](base_features)

    train = pairs[pairs["split"] == "train"]
    validation = pairs[pairs["split"] == "validation"]
    if train.empty or validation.empty:
        raise SystemExit("Train or validation split is empty; check the temporal split manifest.")

    # The locked temporal test split is deliberately never materialized here.
    LOGGER.info(
        "train=%d pairs / %d subjects | validation=%d pairs / %d subjects | features=%d",
        len(train),
        train["RID"].nunique(),
        len(validation),
        validation["RID"].nunique(),
        len(features),
    )

    numeric, categorical = infer_feature_types(pairs, features)
    preprocessor = build_preprocessor(
        numeric,
        categorical,
        config.get("preprocessing", {}).get("numeric_impute", "median"),
        bool(config.get("preprocessing", {}).get("add_missing_indicators", True)),
    )

    if args.mode == "holdout":
        X_train = preprocessor.fit_transform(train[features])
        X_val = preprocessor.transform(validation[features])
        kwargs = {
            "X_val": np.asarray(X_val),
            "y_val": validation["label"].to_numpy(),
            "sample_weight": train["subject_weight"].to_numpy(),
        }
        X_fit = np.asarray(X_train)
        y_fit = train["label"].to_numpy()
    else:
        combined = pairs[pairs["split"].isin(["train", "validation"])]
        X_fit = np.asarray(preprocessor.fit_transform(combined[features]))
        y_fit = combined["label"].to_numpy()
        kwargs = {
            "groups": combined["RID"].to_numpy(),
            "sample_weight": combined["subject_weight"].to_numpy(),
            "n_splits": args.folds,
        }

    summary = tune_many(
        args.models,
        X_fit,
        y_fit,
        output_dir,
        n_trials=args.trials,
        timeout=args.timeout,
        metric=args.metric,
        mode=args.mode,
        use_gpu=args.gpu,
        seed=args.seed,
        log_mlflow=not args.no_mlflow,
        progress=True,
        **kwargs,
    )

    print("\n=== Tuning summary ===")
    print(summary.to_string(index=False))
    tuning_dir = output_dir / "tuning"
    print(f"\nPer-model best params written to {tuning_dir}")

    if not summary.empty:
        best = summary.iloc[0]
        params_path = tuning_dir / f"best_params_{best['model']}.json"
        params = json.loads(params_path.read_text(encoding="utf-8"))["best_params"]
        print(f"\nBest model: {best['model']} ({args.metric}={best['best_value']:.4f})")
        print("Paste into your config under models:")
        print(f"  {best['model']}:")
        for key, value in params.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
