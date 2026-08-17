import json

import pandas as pd
import pytest

from src.tadpole.phase_b_report import generate_phase_b_report


def _workspace(tmp_path, seeds=(1, 2, 3, 4, 5), complete=True):
    phase_a = tmp_path / "outputs" / "phase_a"
    phase_b = tmp_path / "outputs" / "phase_b"
    phase_a.mkdir(parents=True)
    (phase_b / "sparse_modalities").mkdir(parents=True)
    (phase_a / "primary_whitelist.json").write_text(json.dumps(["AGE", "MMSE"]), encoding="utf-8")
    pd.DataFrame({"model": ["random_forest"], "macro_f1_mean": [0.80]}).to_csv(
        phase_b / "baseline_results_summary.csv", index=False
    )
    pd.DataFrame({"seed": list(seeds), "macro_f1": [0.8] * len(seeds)}).to_csv(
        phase_b / "baseline_results_by_seed.csv", index=False
    )
    pd.DataFrame(
        {
            "data_source": ["compact", "full_primary"],
            "model": ["random_forest", "random_forest"],
            "macro_f1_mean": [0.74, 0.81],
        }
    ).to_csv(phase_b / "compact_vs_full_summary.csv", index=False)
    pd.DataFrame({"modality": ["cognitive"], "macro_f1_mean": [0.7]}).to_csv(
        phase_b / "modality_ablation_summary.csv", index=False
    )
    pd.DataFrame({"dropped": ["mri_derived"], "macro_f1_mean": [0.69]}).to_csv(
        phase_b / "missing_modality_summary.csv", index=False
    )
    pd.DataFrame({"feature": ["MMSE"], "importance": [0.4]}).to_csv(phase_b / "feature_importance.csv", index=False)
    if complete:
        (phase_b / "cohort_baseline_only_summary.md").write_text(
            "# Baseline cohort\n\nBaseline cohort body.", encoding="utf-8"
        )
        (phase_b / "cohort_all_visits_summary.md").write_text("All-visit cohort body.", encoding="utf-8")
    return phase_b


def test_report_renders_whitelist_tables_and_cohort_sections(tmp_path, monkeypatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    generate_phase_b_report("outputs/phase_b/final_report.md")

    report = (tmp_path / "outputs" / "phase_b" / "final_report.md").read_text(encoding="utf-8")
    assert "- Frozen primary whitelist: 2 features" in report
    assert "`AGE`, `MMSE`" in report
    assert "| model | macro_f1_mean |" in report
    assert "Baseline cohort body." in report
    assert "# Baseline cohort" not in report
    assert "All-visit cohort body." in report
    assert "Ready for model freezing and D3/D4 external evaluation." in report


def test_pending_placeholders_appear_for_absent_inputs(tmp_path, monkeypatch):
    _workspace(tmp_path, complete=False)
    monkeypatch.chdir(tmp_path)

    generate_phase_b_report("outputs/phase_b/final_report.md")

    report = (tmp_path / "outputs" / "phase_b" / "final_report.md").read_text(encoding="utf-8")
    assert "_Pending: `outputs/phase_b/cohort_baseline_only_summary.md`_" in report
    assert "_Pending._" in report  # modality importance table was never produced
    assert "Not ready for D3/D4 external evaluation" in report
    assert "- `outputs/phase_b/cohort_all_visits_summary.md`" in report


def test_phase_c_readiness_requires_at_least_five_seeds(tmp_path, monkeypatch):
    _workspace(tmp_path, seeds=(1, 2, 3))
    monkeypatch.chdir(tmp_path)

    generate_phase_b_report("outputs/phase_b/final_report.md")

    report = (tmp_path / "outputs" / "phase_b" / "final_report.md").read_text(encoding="utf-8")
    assert "Not ready for D3/D4 external evaluation" in report


def test_comparison_conclusion_reports_the_signed_macro_f1_difference(tmp_path, monkeypatch):
    phase_b = _workspace(tmp_path)
    pd.DataFrame({"data_source": ["compact", "full_primary"], "macro_f1_mean": [0.8100, 0.7400]}).to_csv(
        phase_b / "compact_vs_full_summary.csv", index=False
    )
    monkeypatch.chdir(tmp_path)

    generate_phase_b_report("outputs/phase_b/final_report.md")

    report = (tmp_path / "outputs" / "phase_b" / "final_report.md").read_text(encoding="utf-8")
    assert "does not improve mean Macro F1 by -0.0700" in report


def test_explainability_warning_blocks_report_generation(tmp_path, monkeypatch):
    phase_b = _workspace(tmp_path)
    (phase_b / "EXPLAINABILITY_WARNING").write_text("DX_bl ranked first", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="DX_bl ranked first"):
        generate_phase_b_report("outputs/phase_b/final_report.md")

    assert not (phase_b / "final_report.md").exists()
