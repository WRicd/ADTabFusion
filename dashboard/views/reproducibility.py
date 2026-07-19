from __future__ import annotations

import streamlit as st

from dashboard.artifacts import artifact, load_json, safe_artifact_name
from dashboard.components import artifact_table, empty_state, page_header, section_header, status_badge
from dashboard.i18n import bilingual, get_language


lang = get_language()
page_header(
    bilingual("审计与冻结", "AUDIT AND FREEZE", lang),
    bilingual("可复现性", "Reproducibility", lang),
    bilingual("展示完成标记、冻结注册表和安全化的 artifact 标识。", "Completion markers, frozen registry entries, and sanitized artifact identifiers.", lang),
)
status_badge(bilingual("Phase D 已冻结", "Phase D frozen", lang), "success")

section_header(bilingual("阶段报告", "Phase reports", lang))
reports = [
    {"phase": "A", "purpose": "Audit", "path": "outputs/phase_a/audit_report.md"},
    {"phase": "B", "purpose": "Internal modeling", "path": "outputs/phase_b/compact_vs_full_report.md"},
    {"phase": "C", "purpose": "D3/D4 matching", "path": "outputs/phase_c/evaluation/d3_d4_matching_report.md"},
    {"phase": "D", "purpose": "Frozen final evidence", "path": "outputs/reports/phase_d_final_report.md"},
]
artifact_table(reports)

section_header(bilingual("冻结注册表", "Frozen registry", lang))
registry, error = load_json(artifact("phase_d/manifests/phase_d_frozen_registry.json"))
if error:
    empty_state(error)
else:
    raw_entries = registry.get("artifacts", registry.get("files", []))
    records: list[dict[str, object]] = []
    if isinstance(raw_entries, dict):
        for name, value in raw_entries.items():
            if isinstance(value, dict):
                records.append({"artifact": name, "path": value.get("path", name), "sha256": value.get("sha256", value.get("hash", ""))})
            else:
                records.append({"artifact": name, "path": safe_artifact_name(str(value))})
    elif isinstance(raw_entries, list):
        for index, value in enumerate(raw_entries):
            if isinstance(value, dict):
                records.append({"artifact": value.get("name", f"artifact_{index + 1}"), "path": value.get("path", ""), "sha256": value.get("sha256", value.get("hash", ""))})
    if records:
        artifact_table(records)
    else:
        st.json({key: value for key, value in registry.items() if "path" not in key.lower()}, expanded=False)

section_header(bilingual("本地运行", "Local launch", lang))
st.code("streamlit run dashboard/app.py", language="bash")
st.write(
    bilingual(
        "看板仅读取 outputs 中已有 CSV/JSON；数据或 artifact 缺失时显示空状态。",
        "The dashboard reads existing CSV/JSON files under outputs only and shows an empty state when an artifact is missing.",
        lang,
    )
)
