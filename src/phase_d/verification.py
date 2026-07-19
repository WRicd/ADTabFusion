from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.external.d3_cohort import select_d3_index_records
from src.external.inference import predict_with_frozen_pipeline, probability_frame
from src.external.model_freezing import sha256_file
from src.external.schema_alignment import align_to_frozen_schema


def verify_phase_c_artifacts(root: str | Path = "outputs/phase_c") -> dict[str, Any]:
    root = Path(root)
    checks: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {}
    specs = [
        ("primary", root / "models/primary_pipeline.joblib", root / "manifests/primary_model_manifest.json", Path("configs/phase_c_direct_transfer.yaml")),
        ("sensitivity", root / "models/sensitivity_pipeline.joblib", root / "manifests/sensitivity_model_manifest.json", Path("configs/phase_c_direct_transfer.yaml")),
        ("horizon_aware", root / "models/horizon_aware_pipeline.joblib", root / "manifests/horizon_aware_manifest.json", Path("configs/phase_c_horizon_aware.yaml")),
    ]
    manifests = {}
    for role, model_path, manifest_path, config_path in specs:
        _check(checks, f"{role}_model_exists", model_path.exists(), str(model_path))
        _check(checks, f"{role}_manifest_exists", manifest_path.exists(), str(manifest_path))
        if not model_path.exists() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[role] = manifest
        model_hash = sha256_file(model_path)
        config_hash = sha256_file(config_path)
        hashes[role] = {"model_sha256": model_hash, "config_sha256": config_hash}
        _check(checks, f"{role}_model_hash", model_hash == manifest["model_sha256"], model_hash)
        _check(checks, f"{role}_config_hash", config_hash == manifest["config_sha256"], config_hash)
        pipeline = joblib.load(model_path)
        fitted_order = list(pipeline.named_steps["preprocessor"].feature_names_in_)
        _check(checks, f"{role}_feature_order", fitted_order == manifest["feature_order"], f"{len(fitted_order)} features")
    d3_path = Path("data/tadpole_challenge/TADPOLE_D3.csv")
    direct_manifests = {role: manifests[role] for role in ("primary", "sensitivity") if role in manifests}
    if direct_manifests:
        features = list(dict.fromkeys(feature for item in direct_manifests.values() for feature in item["feature_order"]))
        required = ["RID", "EXAMDATE", *features]
        raw = pd.read_csv(d3_path, usecols=lambda column: column in required, low_memory=False, na_values=["", " ", "-4", "-4.0"])
        cohort, _ = select_d3_index_records(raw)
        for role in direct_manifests:
            reproduced, _ = predict_with_frozen_pipeline(
                root / f"models/{role}_pipeline.joblib", root / f"manifests/{role}_model_manifest.json", cohort
            )
            stored = pd.read_csv(root / f"predictions/d3_direct_{role}.csv")
            probability_columns = ["prob_CN", "prob_MCI", "prob_AD"]
            same = np.allclose(reproduced[probability_columns], stored[probability_columns], atol=1e-12)
            _check(checks, f"{role}_d3_reproduced", same, f"{len(stored)} rows")
            _check(checks, f"{role}_probability_simplex", np.allclose(stored[probability_columns].sum(axis=1), 1.0, atol=1e-8), "sum=1")
    matched = pd.read_csv(root / "evaluation/d3_d4_matched_rows.csv", low_memory=False)
    _check(checks, "matched_rows", len(matched) == 210, str(len(matched)))
    _check(checks, "matched_subjects", matched["RID"].nunique() == 197, str(matched["RID"].nunique()))
    first = matched.sort_values(["RID", "D4_SCANDATE"]).drop_duplicates("RID")
    _check(checks, "first_follow_up_unique", len(first) == first["RID"].nunique() == 197, str(len(first)))
    if "horizon_aware" in manifests:
        pipeline = joblib.load(root / "models/horizon_aware_pipeline.joblib")
        aligned = align_to_frozen_schema(matched, manifests["horizon_aware"]["feature_order"])
        reproduced = probability_frame(pipeline.predict_proba(aligned))
        stored = pd.read_csv(root / "predictions/d3_d4_horizon_aware.csv")
        columns = ["prob_CN", "prob_MCI", "prob_AD"]
        _check(checks, "horizon_d3_d4_reproduced", np.allclose(reproduced[columns], stored[columns], atol=1e-12), f"{len(stored)} rows")
    eval_paths = [root / "evaluation/direct_transfer_metrics.csv", root / "evaluation/horizon_aware_metrics.csv"]
    latest_model = max((path.stat().st_mtime for _, path, _, _ in specs if path.exists()), default=0)
    order_ok = all(path.exists() and path.stat().st_mtime >= latest_model for path in eval_paths)
    _check(checks, "artifact_creation_order", order_ok, "evaluation created after frozen models")
    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "checks": checks, "hashes": hashes}


def write_verification_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase_c_artifact_hashes.json").write_text(
        json.dumps(result["hashes"], indent=2), encoding="utf-8"
    )
    lines = ["# Phase C Verification Gate", "", f"Overall status: **{'PASS' if result['passed'] else 'FAIL'}**", "", "| Check | Status | Detail |", "|---|---|---|"]
    for item in result["checks"]:
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |")
    lines.append("")
    (output / "phase_c_verification_report.md").write_text("\n".join(lines), encoding="utf-8")


def _check(checks: list[dict[str, Any]], name: str, passed: Any, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
