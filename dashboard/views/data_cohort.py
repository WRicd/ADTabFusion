from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.artifacts import artifact, load_csv, load_json
from dashboard.charts import show_chart
from dashboard.components import empty_state, limitation_banner, metric_card, page_header, section_header, status_badge
from dashboard.i18n import bilingual, get_language
from dashboard.theme import COLORS


lang = get_language()
page_header(
    bilingual("聚合数据概览", "AGGREGATE DATA OVERVIEW", lang),
    bilingual("数据与队列", "Data & Cohort", lang),
    bilingual("仅展示数据规模、模态可用性、时间划分与诊断转归计数。", "Aggregate dataset size, modality availability, temporal split, and transition counts only.", lang),
)
status_badge(bilingual("无参与者级展示", "No participant-level display", lang), "success")

stats, stats_error = load_json(artifact("phase_d/cohorts/transition_pair_statistics.json"))
manifest, manifest_error = load_json(artifact("phase_d/cohorts/temporal_split_manifest.json"))

top = st.columns(4)
with top[0]:
    metric_card(bilingual("纵向受试者", "Longitudinal subjects", lang), f"{stats.get('n_subjects', stats.get('subjects', 1611)):,}", bilingual("转归配对队列", "Transition-pair cohort", lang))
with top[1]:
    metric_card(bilingual("预测配对", "Prediction pairs", lang), f"{stats.get('n_pairs', stats.get('total_pairs', 19073)):,}", "6–60 months")
with top[2]:
    metric_card(bilingual("锁定测试受试者", "Locked test subjects", lang), "262", bilingual("按时间切分", "Temporal split", lang))
with top[3]:
    metric_card(bilingual("锁定测试配对", "Locked test pairs", lang), "893", bilingual("主要结果队列", "Primary result cohort", lang))

section_header(bilingual("可用模态", "Available modalities", lang))
modalities = pd.DataFrame(
    [
        {"Modality": "Demographic", "Status": "Active"},
        {"Modality": "Cognitive / clinical", "Status": "Active"},
        {"Modality": "MRI-derived", "Status": "Active"},
        {"Modality": "APOE4", "Status": "Active"},
        {"Modality": "PET", "Status": "Sparse / cohort analysis"},
        {"Modality": "CSF", "Status": "Sparse / cohort analysis"},
        {"Modality": "DTI", "Status": "Sparse / cohort analysis"},
    ]
)
colors = alt.Scale(domain=["Active", "Sparse / cohort analysis"], range=[COLORS["teal"], COLORS["amber"]])
chart = (
    alt.Chart(modalities)
    .mark_bar(cornerRadiusEnd=3, size=22)
    .encode(
        x=alt.X("Status:N", title=None, axis=None),
        y=alt.Y("Modality:N", title=None, sort=modalities["Modality"].tolist()),
        color=alt.Color("Status:N", scale=colors, legend=alt.Legend(orient="top")),
        tooltip=["Modality:N", "Status:N"],
    )
    .properties(height=270)
)
show_chart(chart)

section_header(bilingual("诊断转归结构", "Diagnosis transition structure", lang))
matrix, matrix_error = load_csv(artifact("phase_d/internal_validation/transition_matrix.csv"), ("split", "SOURCE_DX", "FUTURE_DX", "count"))
if matrix_error:
    empty_state(matrix_error)
else:
    test_matrix = matrix.loc[matrix["split"] == "temporal_test"].copy()
    heat = (
        alt.Chart(test_matrix)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("FUTURE_DX:N", title="Future diagnosis", sort=["CN", "MCI", "AD"]),
            y=alt.Y("SOURCE_DX:N", title="Source diagnosis", sort=["CN", "MCI", "AD"]),
            color=alt.Color("count:Q", scale=alt.Scale(range=["#EAF2F7", COLORS["blue"]]), legend=None),
            tooltip=["SOURCE_DX:N", "FUTURE_DX:N", "count:Q"],
        )
        .properties(height=290)
    )
    labels = heat.mark_text().encode(text="count:Q", color=alt.condition("datum.count > 160", alt.value("white"), alt.value(COLORS["ink"])))
    show_chart(heat + labels)

if stats_error or manifest_error:
    limitation_banner(bilingual("部分队列元数据不可用，但结果页面仍可读取冻结评估 CSV。", "Some cohort metadata is unavailable; frozen evaluation CSVs remain readable.", lang))

limitation_banner(
    bilingual(
        "稀疏 PET、CSF 与 DTI 不能视为主模型中的完整活跃模态；页面不展示 RID、PTID 或行级记录。",
        "Sparse PET, CSF, and DTI are not complete active modalities in the primary model; RID, PTID, and row-level records are never shown.",
        lang,
    )
)
