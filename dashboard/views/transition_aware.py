from __future__ import annotations

import ast

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.artifacts import artifact, load_csv, load_json
from dashboard.charts import show_chart
from dashboard.components import chart_heading, empty_state, interpretation_box, metric_card, page_header, section_header, status_badge
from dashboard.i18n import bilingual, get_language
from dashboard.theme import CLASS_COLORS, CLASS_ORDER, COLORS


lang = get_language()
page_header(
    bilingual("主要模型", "PRIMARY MODEL", lang),
    bilingual("转归感知模型", "Transition-Aware Model", lang),
    bilingual("预测 6–60 个月未来诊断状态的纵向多分类模型。", "Longitudinal multiclass prediction of future diagnosis 6–60 months ahead.", lang),
)
status_badge(bilingual("锁定时间测试", "Locked temporal test", lang), "success")

results, error = load_csv(artifact("phase_d/temporal_validation/transition_model_results.csv"))
row = results.loc[results.get("split", pd.Series(dtype=str)) == "locked_temporal_test"] if not results.empty else pd.DataFrame()
if error or row.empty:
    empty_state(error or "Transition metrics are unavailable.")
else:
    result = row.iloc[0]
    cards = st.columns(5)
    values = [
        ("Accuracy", result["accuracy"]),
        ("Balanced Accuracy", result["balanced_accuracy"]),
        ("Macro F1", result["macro_f1"]),
        ("ROC-AUC OvR", result["roc_auc_ovr"]),
        ("Log Loss", result["log_loss"]),
    ]
    for column, (label, value) in zip(cards, values):
        with column:
            metric_card(label, f"{value:.3f}", "locked temporal test")

    section_header(bilingual("分类表现", "Class-level performance", lang))
    class_rows = []
    for index, label in enumerate(CLASS_ORDER):
        class_rows.append(
            {
                "Class": label,
                "Precision": result[f"precision_class_{index}"],
                "Recall": result[f"recall_class_{index}"],
                "F1": result[f"f1_class_{index}"],
            }
        )
    class_frame = pd.DataFrame(class_rows).melt("Class", var_name="Metric", value_name="Score")
    class_chart = (
        alt.Chart(class_frame)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("Metric:N", title=None),
            xOffset="Class:N",
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Class:N", scale=alt.Scale(domain=CLASS_ORDER, range=[CLASS_COLORS[x] for x in CLASS_ORDER]), legend=alt.Legend(orient="top")),
            tooltip=["Class:N", "Metric:N", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=300)
    )
    show_chart(class_chart)

    section_header(bilingual("锁定测试混淆矩阵", "Locked test confusion matrix", lang))
    matrix = ast.literal_eval(str(result["confusion_matrix"]))
    heat_rows = [
        {"Actual": actual, "Predicted": predicted, "Count": matrix[i][j]}
        for i, actual in enumerate(CLASS_ORDER)
        for j, predicted in enumerate(CLASS_ORDER)
    ]
    heat = (
        alt.Chart(pd.DataFrame(heat_rows))
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("Predicted:N", sort=CLASS_ORDER),
            y=alt.Y("Actual:N", sort=CLASS_ORDER),
            color=alt.Color("Count:Q", scale=alt.Scale(range=["#EAF2F7", COLORS["blue"]]), legend=None),
            tooltip=["Actual:N", "Predicted:N", "Count:Q"],
        )
        .properties(height=260)
    )
    text = heat.mark_text().encode(text="Count:Q", color=alt.condition("datum.Count > 160", alt.value("white"), alt.value(COLORS["ink"])))
    show_chart(heat + text)

section_header(bilingual("消融证据", "Ablation evidence", lang))
ablation, ablation_error = load_csv(artifact("phase_d/internal_validation/transition_ablation.csv"))
if ablation_error:
    empty_state(ablation_error)
else:
    label_map = {
        "features_only": "Features only",
        "features_plus_forecast": "+ Horizon",
        "features_plus_source_dx": "+ Source diagnosis",
        "features_plus_source_dx_forecast": "+ Source diagnosis + horizon",
    }
    chart_frame = ablation.assign(Label=ablation["ablation"].map(label_map))
    chart_heading("Validation Macro F1", bilingual("比较固定验证集上的四种输入组合。", "Four input combinations on the fixed validation split.", lang))
    chart = (
        alt.Chart(chart_frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("macro_f1:Q", scale=alt.Scale(domain=[0, 1]), title="Macro F1"),
            y=alt.Y("Label:N", sort=list(label_map.values()), title=None),
            color=alt.condition("datum.ablation === 'features_plus_source_dx_forecast'", alt.value(COLORS["teal"]), alt.value(COLORS["blue"])),
            tooltip=["Label:N", alt.Tooltip("macro_f1:Q", format=".3f")],
        )
        .properties(height=260)
    )
    show_chart(chart)
    interpretation_box(
        bilingual(
            "加入当前诊断状态后 Macro F1 从 0.645 提升到 0.810；再加入预测时间跨度后达到 0.820。",
            "Adding source diagnosis lifts Macro F1 from 0.645 to 0.810; adding forecast horizon reaches 0.820.",
            lang,
        )
    )

stats, stats_error = load_json(artifact("phase_d/cohorts/transition_pair_statistics.json"))
if not stats_error and stats:
    with st.expander(bilingual("队列摘要", "Cohort summary", lang)):
        st.write(stats)
