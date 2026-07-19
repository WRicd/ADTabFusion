from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


OUTPUT_DIR = Path("outputs")

TEXT = {
    "language_label": {"en": "Language", "zh": "语言"},
    "english": {"en": "English", "zh": "English"},
    "chinese": {"en": "中文", "zh": "中文"},
    "overview": {"en": "Overview", "zh": "总览"},
    "raw_inventory": {"en": "Raw Data Inventory", "zh": "原始数据盘点"},
    "scanned_csvs": {"en": "Scanned CSVs", "zh": "已扫描 CSV"},
    "available_categories": {"en": "Available Categories", "zh": "可用数据类别"},
    "missing_categories": {"en": "Missing Categories", "zh": "缺失数据类别"},
    "next_downloads": {"en": "Recommended next downloads", "zh": "建议下一步下载"},
    "inventory_missing": {
        "en": "ADNI inventory has not been generated. Run `python scripts/inspect_raw_adni_csvs.py --raw-dir data/raw --output-md docs/adni_file_inventory.md`.",
        "zh": "尚未生成 ADNI 数据盘点。请运行 `python scripts/inspect_raw_adni_csvs.py --raw-dir data/raw --output-md docs/adni_file_inventory.md`。",
    },
    "no_download_needed": {
        "en": "No missing second-batch modality category was detected.",
        "zh": "未检测到缺失的第二批模态类别。",
    },
    "rows": {"en": "Rows", "zh": "样本行数"},
    "subjects": {"en": "Subjects", "zh": "受试者数"},
    "columns": {"en": "Columns", "zh": "字段数"},
    "active_modalities": {"en": "Active modalities:", "zh": "当前可用模态："},
    "unavailable_modalities": {"en": "Unavailable modalities:", "zh": "当前不可用模态："},
    "not_generated": {"en": "Not generated yet", "zh": "尚未生成"},
    "none_detected": {"en": "None detected", "zh": "未检测到"},
    "task_modes": {"en": "Task modes: `baseline_only`, `all_visits`", "zh": "任务模式：`baseline_only`、`all_visits`"},
    "dashboard_info": {
        "en": "Dashboard reads existing files from outputs/ and does not retrain models.",
        "zh": "数据看板只读取 outputs/ 中已有结果，不会重新训练模型。",
    },
    "limitations": {"en": "Limitations", "zh": "当前限制"},
    "limitations_body": {
        "en": "- Existing model outputs still use the D1_D2 feature table.\n- Raw PET and biofluid CSVs are detected but have not been merged into model training.\n- An independent data dictionary and D3 external test data are not currently available.\n- Raw-image processing, GNN, Transformer, OASIS-3, and cloud deployment are out of scope.",
        "zh": "- 现有模型结果仍使用 D1_D2 特征表。\n- 已检测到 PET 和体液 CSV，但尚未合并进模型训练。\n- 当前缺少独立数据字典和 D3 外部测试数据。\n- 原始影像处理、GNN、Transformer、OASIS-3 和云端部署不在当前阶段范围内。",
    },
    "data_audit": {"en": "Data Audit", "zh": "数据审计"},
    "audit_missing": {"en": "Data audit not found. Run prepare_tadpole first.", "zh": "未找到数据审计结果。请先运行 prepare_tadpole。"},
    "missingness": {"en": "Missingness", "zh": "缺失率"},
    "model_performance": {"en": "Model Performance", "zh": "模型性能"},
    "seed_summary": {"en": "Seed Summary", "zh": "多随机种子汇总"},
    "by_seed": {"en": "By Seed", "zh": "按随机种子结果"},
    "confusion_matrix": {"en": "Confusion matrix", "zh": "混淆矩阵"},
    "modality_ablation": {"en": "Modality Ablation", "zh": "模态消融"},
    "missing_modality": {"en": "Missing Modality Robustness", "zh": "缺失模态鲁棒性"},
    "explainability": {"en": "Explainability", "zh": "可解释性分析"},
    "top_features": {"en": "Top Features", "zh": "重要特征"},
    "modality_importance": {"en": "Modality Importance", "zh": "模态重要性"},
    "error_cases": {"en": "Error Cases", "zh": "错误案例"},
    "high_confidence_errors": {"en": "High-confidence Errors", "zh": "高置信错误样本"},
    "download_error_cases": {"en": "Download error cases", "zh": "下载错误案例 CSV"},
    "per_class_confusion": {"en": "Per-class Confusion Summary", "zh": "按类别混淆汇总"},
    "empty_table": {"en": "No table found. Run the pipeline first.", "zh": "未找到结果表。请先运行实验流水线。"},
    "phase_d_overview": {"en": "Phase D Progression Modeling", "zh": "Phase D 疾病进展建模"},
    "primary_temporal_result": {"en": "Primary: Locked D1/D2 Temporal Test", "zh": "主要结果：D1/D2 锁定时间测试集"},
    "exploratory_replay": {"en": "Secondary: Exploratory Post-Hoc D4 Replay", "zh": "次要结果：探索性 D4 事后回放"},
    "d3_features": {"en": "D3-Compatible Features", "zh": "D3 兼容特征"},
    "transition_matrix": {"en": "Transition Matrix", "zh": "诊断转化矩阵"},
    "mci_conversion": {"en": "MCI Conversion Risk", "zh": "MCI 转 AD 风险"},
    "calibration": {"en": "Probability Calibration", "zh": "概率校准"},
    "risk_coverage": {"en": "Risk-Coverage", "zh": "风险-覆盖率"},
    "temporal_validation": {"en": "Temporal Validation", "zh": "时间验证"},
    "d4_replay": {"en": "D4 Exploratory Replay", "zh": "D4 探索性回放"},
    "primary_notice": {"en": "Primary scientific result. Model decisions use D1/D2 validation subjects; this table is the locked temporal test.", "zh": "主要科研结果。模型决策仅使用 D1/D2 验证 subjects；本表为锁定时间测试集。"},
    "replay_notice": {"en": "Exploratory post-hoc replay only. D4 was previously accessed in Phase C and is not independent confirmatory validation.", "zh": "仅为探索性事后回放。D4 已在 Phase C 使用，不属于独立确认性验证。"},
}

MODALITY_LABELS = {
    "data_dictionary": {"en": "data dictionary", "zh": "数据字典"},
    "diagnosis": {"en": "diagnosis", "zh": "诊断"},
    "demographic": {"en": "demographic", "zh": "人口学"},
    "cognitive": {"en": "cognitive", "zh": "认知量表"},
    "mri_derived": {"en": "MRI-derived", "zh": "MRI 派生特征"},
    "genetic": {"en": "APOE4/genetic", "zh": "APOE4/遗传"},
    "apoe": {"en": "APOE", "zh": "APOE"},
    "pet": {"en": "PET", "zh": "PET"},
    "csf": {"en": "CSF", "zh": "CSF"},
    "mri_measurement": {"en": "MRI measurements", "zh": "MRI 测量"},
    "pet_measurement": {"en": "PET measurements", "zh": "PET 测量"},
    "biofluid_csf": {"en": "biofluid / CSF", "zh": "体液 / CSF"},
}


def read_json(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def available_modalities() -> tuple[list[str], list[str]]:
    used = read_json(OUTPUT_DIR / "metrics" / "used_features.json")
    groups = used.get("groups", {})
    active = [name for name, cols in groups.items() if cols]
    unavailable = []
    for name in ["pet", "csf"]:
        if name not in groups or not groups.get(name):
            unavailable.append(name)
    return active, unavailable


def raw_inventory_summary(output_dir: str | Path = OUTPUT_DIR) -> dict:
    """Summarize generated raw-file inventory data for the dashboard."""
    root = Path(output_dir)
    inventory = read_json(root / "metrics" / "adni_file_inventory.json")
    availability = read_json(root / "metrics" / "adni_modality_availability.json")
    available = [
        key for key, value in availability.items() if value.get("available", False)
    ]
    missing = [
        key for key, value in availability.items() if not value.get("available", False)
    ]
    return {
        "scanned_csvs": len(inventory) if isinstance(inventory, list) else 0,
        "available": available,
        "missing": missing,
        "generated": bool(inventory) or bool(availability),
    }


def inventory_download_recommendations(availability: dict, lang: str) -> list[str]:
    routes = {
        "pet_measurement": {
            "en": "Study Files -> Imaging -> PET Image Analysis",
            "zh": "Study Files -> Imaging -> PET Image Analysis（PET 影像分析）",
        },
        "biofluid_csf": {
            "en": "Biospecimen -> Biospecimen Results",
            "zh": "Biospecimen -> Biospecimen Results（生物样本结果）",
        },
        "mri_measurement": {
            "en": "Study Files -> Imaging -> MR Image Analysis",
            "zh": "Study Files -> Imaging -> MR Image Analysis（MR 影像分析）",
        },
        "cognitive": {
            "en": "Study Files -> Assessments -> Neuropsychological",
            "zh": "Study Files -> Assessments -> Neuropsychological（神经心理评估）",
        },
    }
    return [
        labels[lang]
        for key, labels in routes.items()
        if not availability.get(key, {}).get("available", False)
    ]


def image_path(name: str) -> Path:
    return OUTPUT_DIR / "figures" / name


def language_selector() -> str:
    """Render a shared language selector and return the current language key."""
    current = st.session_state.get("language", "zh")
    options = ["zh", "en"]
    labels = {"zh": TEXT["chinese"][current], "en": TEXT["english"][current]}
    selected = st.sidebar.selectbox(
        TEXT["language_label"][current],
        options,
        index=options.index(current),
        format_func=lambda key: labels[key],
        key="language",
    )
    return selected


def tr(key: str, lang: str) -> str:
    """Translate a fixed dashboard text key."""
    return TEXT.get(key, {}).get(lang, key)


def modality_label(name: str, lang: str) -> str:
    """Translate modality names used in metrics outputs."""
    return MODALITY_LABELS.get(name, {}).get(lang, name)


def display_modalities(names: list[str], lang: str) -> str:
    """Return a comma-separated modality label string."""
    return ", ".join(modality_label(name, lang) for name in names)


def show_dataframe_or_warning(df: pd.DataFrame, lang: str) -> None:
    """Show a dataframe or a localized warning if it is empty."""
    if df.empty:
        st.warning(tr("empty_table", lang))
    else:
        st.dataframe(df, use_container_width=True)
