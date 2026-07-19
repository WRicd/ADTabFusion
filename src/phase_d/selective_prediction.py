from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation import compute_metrics
from src.phase_d.transition_model import ABLATIONS, load_transition_data


def uncertainty_measures(probability: np.ndarray) -> pd.DataFrame:
    sorted_probability = np.sort(probability, axis=1)
    clipped = np.clip(probability, 1e-15, 1.0)
    return pd.DataFrame(
        {"max_probability": probability.max(axis=1),
         "predictive_entropy": -(clipped * np.log(clipped)).sum(axis=1),
         "probability_margin": sorted_probability[:, -1] - sorted_probability[:, -2]}
    )


def thresholds_from_validation(probability: np.ndarray, coverages: list[float]) -> dict[float, float]:
    confidence = probability.max(axis=1)
    return {float(coverage): float(np.quantile(confidence, 1.0 - coverage)) for coverage in coverages}


def risk_coverage_rows(
    y_true: np.ndarray, probability: np.ndarray, thresholds: dict[float, float], split: str
) -> list[dict[str, Any]]:
    confidence = probability.max(axis=1); rows=[]
    for target_coverage, threshold in thresholds.items():
        retained = confidence >= threshold
        if not retained.any(): continue
        metrics = compute_metrics(y_true[retained], probability[retained].argmax(axis=1), probability[retained], labels=[0,1,2])
        rows.append({"split": split, "target_coverage": target_coverage, "coverage": float(retained.mean()), "threshold": threshold,
                     "n_retained": int(retained.sum()), "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
                     "macro_f1": metrics["macro_f1"], "error_rate": 1.0-metrics["accuracy"]})
    return rows


def evaluate_selective_prediction(config: dict[str, Any]) -> dict[str, Any]:
    output=Path(config["project"]["output_dir"])
    from src.config import load_config
    transition_config=load_config(config["transition_config"])
    pairs,base_features,_=load_transition_data(transition_config);features=ABLATIONS["features_plus_source_dx_forecast"](base_features)
    validation=pairs[pairs.split=="validation"];test=pairs[pairs.split=="temporal_test"]
    model=joblib.load(output/"models"/"calibrated_transition_pipeline.joblib")
    validation_probability=model.predict_proba(validation[features]);test_probability=model.predict_proba(test[features])
    thresholds=thresholds_from_validation(validation_probability,config["coverages"])
    rows=risk_coverage_rows(validation.label.to_numpy(),validation_probability,thresholds,"validation")+risk_coverage_rows(test.label.to_numpy(),test_probability,thresholds,"locked_temporal_test")
    result=pd.DataFrame(rows)
    minimum=config["threshold_selection"].get("minimum_coverage",.7)
    eligible=result[(result.split=="validation")&(result.coverage>=minimum)]
    selected=eligible.sort_values(["macro_f1","coverage"],ascending=[False,False]).iloc[0]
    selected_threshold=float(selected.threshold)
    (output/"uncertainty").mkdir(parents=True,exist_ok=True)
    result["selected_validation_threshold"]=result.threshold.eq(selected_threshold)
    result.to_csv(output/"uncertainty"/"selective_prediction.csv",index=False)
    measures=uncertainty_measures(test_probability)
    pd.DataFrame({"measure":measures.columns,"mean":measures.mean().values,"std":measures.std().values}).to_csv(output/"uncertainty"/"uncertainty_summary.csv",index=False)
    manifest={"threshold":selected_threshold,"selection_split":"D1/D2 validation subjects only","minimum_coverage":minimum,
              "selection_metric":"macro_f1","temporal_test_used_for_threshold_selection":False,"d3_or_d4_used_for_threshold_selection":False}
    (output/"manifests"/"selective_prediction_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    _plot_risk_coverage(result,output/"figures"/"risk_coverage_curve.png")
    return {"manifest":manifest,"results":result}


def _plot_risk_coverage(frame:pd.DataFrame,path:Path)->None:
    import os
    os.environ.setdefault("MPLCONFIGDIR",str(path.parent/".matplotlib"))
    import matplotlib;matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12.8,7.2))
    for split,group in frame.groupby("split"):
        group=group.sort_values("coverage");ax.plot(group.coverage,group.error_rate,marker="o",label=split.replace("_"," ").title())
    ax.set_xlabel("Coverage");ax.set_ylabel("Error Rate");ax.set_title("Selective Prediction Risk-Coverage");ax.legend();fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
