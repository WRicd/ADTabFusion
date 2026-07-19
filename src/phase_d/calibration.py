from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.evaluation import compute_metrics
from src.external.model_freezing import sha256_file, stable_subject_hash
from src.phase_d.transition_model import ABLATIONS, load_transition_data


class ProbabilityCalibratedClassifier:
    def __init__(self, base_estimator, method: str = "uncalibrated"):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrators: list[Any] = []

    def fit_calibration(self, X_validation: pd.DataFrame, y_validation: np.ndarray):
        probability = self.base_estimator.predict_proba(X_validation)
        self.calibrators = []
        if self.method == "uncalibrated":
            return self
        for class_id in range(probability.shape[1]):
            target = (y_validation == class_id).astype(int)
            if self.method == "sigmoid":
                calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(probability[:, [class_id]], target)
            elif self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip").fit(probability[:, class_id], target)
            else:
                raise ValueError(f"Unsupported calibration method: {self.method}")
            self.calibrators.append(calibrator)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base = self.base_estimator.predict_proba(X)
        if self.method == "uncalibrated":
            return base
        columns = []
        for class_id, calibrator in enumerate(self.calibrators):
            if self.method == "sigmoid":
                columns.append(calibrator.predict_proba(base[:, [class_id]])[:, 1])
            else:
                columns.append(calibrator.predict(base[:, class_id]))
        calibrated = np.column_stack(columns)
        total = calibrated.sum(axis=1, keepdims=True)
        zero = total[:, 0] <= 0
        calibrated[zero] = base[zero]
        total = calibrated.sum(axis=1, keepdims=True)
        return calibrated / total

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    confidence = probability.max(axis=1)
    predicted = probability.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classwise_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> dict[str, float]:
    result = {}
    for class_id, name in enumerate(("CN", "MCI", "AD")):
        target = (y_true == class_id).astype(float)
        score = probability[:, class_id]
        edges = np.linspace(0, 1, bins + 1)
        error = 0.0
        for index in range(bins):
            mask = (score > edges[index]) & (score <= edges[index + 1])
            if mask.any(): error += mask.mean() * abs(target[mask].mean() - score[mask].mean())
        result[name] = float(error)
    return result


def calibrate_phase_d(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["project"]["output_dir"])
    transition_config_path = Path(config["transition_config"])
    from src.config import load_config
    transition_config = load_config(transition_config_path)
    pairs, base_features, _ = load_transition_data(transition_config)
    features = ABLATIONS["features_plus_source_dx_forecast"](base_features)
    validation = pairs[pairs["split"] == "validation"].copy()
    temporal_test = pairs[pairs["split"] == "temporal_test"].copy()
    base_path = output / "models" / "transition_aware_pipeline.joblib"
    base = joblib.load(base_path)
    candidates, rows = {}, []
    for method in config["methods"]:
        calibrated = ProbabilityCalibratedClassifier(base, method).fit_calibration(validation[features], validation["label"].to_numpy())
        candidates[method] = calibrated
        for split_name, frame in (("validation", validation), ("locked_temporal_test", temporal_test)):
            probability = calibrated.predict_proba(frame[features])
            metrics = compute_metrics(frame["label"].to_numpy(), probability.argmax(axis=1), probability, labels=[0, 1, 2])
            classwise = classwise_calibration_error(frame["label"].to_numpy(), probability, config.get("ece_bins", 10))
            rows.append(
                {"method": method, "split": split_name, "n_rows": len(frame),
                 "log_loss": metrics["log_loss"], "brier_score": metrics["brier_score"],
                 "macro_f1": metrics["macro_f1"], "roc_auc_ovr": metrics["roc_auc_ovr"],
                 "ece": expected_calibration_error(frame["label"].to_numpy(), probability, config.get("ece_bins", 10)),
                 **{f"classwise_ece_{key}": value for key, value in classwise.items()}},
            )
    results = pd.DataFrame(rows)
    validation_results = results[results["split"] == "validation"]
    selected_method = str(validation_results.sort_values(["log_loss", "brier_score"]).iloc[0]["method"])
    selected = candidates[selected_method]
    (output / "calibration").mkdir(parents=True, exist_ok=True)
    results["selected_on_validation"] = results["method"].eq(selected_method)
    results.to_csv(output / "calibration" / "calibration_results.csv", index=False)
    model_path = output / "models" / "calibrated_transition_pipeline.joblib"
    joblib.dump(selected, model_path)
    manifest = {
        "model_id": f"phase_d_calibrated_transition_{selected_method}",
        "calibration_method": selected_method, "feature_order": features,
        "calibration_split": "D1/D2 validation subjects only",
        "calibration_subject_count": int(validation["RID"].nunique()),
        "calibration_subject_rid_sha256": stable_subject_hash(validation["RID"]),
        "base_model_sha256": sha256_file(base_path), "calibrated_model_sha256": sha256_file(model_path),
        "temporal_test_used_for_calibration_or_selection": False,
        "d3_or_d4_used_for_calibration_or_selection": False,
    }
    (output / "manifests" / "calibrated_transition_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _plot_reliability(base, selected, temporal_test, features, output / "figures" / "reliability_diagram.png")
    return {"manifest": manifest, "results": results}


def _plot_reliability(base, calibrated, frame: pd.DataFrame, features: list[str], path: Path) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib; matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharex=True, sharey=True)
    truth = frame["label"].to_numpy()
    for class_id, name in enumerate(("CN", "MCI", "AD")):
        for model, label, color in ((base, "Uncalibrated", "#4c78a8"), (calibrated, "Calibrated", "#e15759")):
            score = model.predict_proba(frame[features])[:, class_id]
            bins = np.minimum((score * 10).astype(int), 9); x=[]; y=[]
            for bin_id in range(10):
                mask=bins==bin_id
                if mask.any(): x.append(score[mask].mean()); y.append((truth[mask]==class_id).mean())
            axes[class_id].plot(x,y,marker="o",label=label,color=color)
        axes[class_id].plot([0,1],[0,1],"--",color="gray"); axes[class_id].set_title(name); axes[class_id].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed frequency"); axes[-1].legend(); fig.suptitle("Temporal-Test Reliability by Class"); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
