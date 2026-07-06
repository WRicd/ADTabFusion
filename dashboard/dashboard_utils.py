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
        "en": "- This version uses only D1_D2.csv.\n- PET/CSF/D3 are not available in the current local data.\n- Raw MRI, GNN, Transformer, OASIS-3, and cloud deployment are intentionally out of scope.",
        "zh": "- 当前版本仅使用 D1_D2.csv。\n- 当前本地数据不包含 PET、CSF、D3。\n- raw MRI、GNN、Transformer、OASIS-3 和云端部署不在当前阶段范围内。",
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
}

MODALITY_LABELS = {
    "demographic": {"en": "demographic", "zh": "人口学"},
    "cognitive": {"en": "cognitive", "zh": "认知量表"},
    "mri_derived": {"en": "MRI-derived", "zh": "MRI 派生特征"},
    "genetic": {"en": "APOE4/genetic", "zh": "APOE4/遗传"},
    "pet": {"en": "PET", "zh": "PET"},
    "csf": {"en": "CSF", "zh": "CSF"},
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
