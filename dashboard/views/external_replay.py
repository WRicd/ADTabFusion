from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.artifacts import artifact, load_csv
from dashboard.charts import show_chart
from dashboard.components import empty_state, interpretation_box, limitation_banner, metric_card, page_header, section_header, status_badge
from dashboard.i18n import bilingual, get_language
from dashboard.theme import COLORS


EXPLORATORY_LABEL = "Exploratory post-hoc replay — not independent confirmatory validation"

lang = get_language()
page_header(
    "D4",
    bilingual("外部回放", "External Replay", lang),
    EXPLORATORY_LABEL,
)
status_badge(EXPLORATORY_LABEL, "warning")
limitation_banner(
    bilingual(
        "该分析在 D4 标签可用后进行，仅用于探索性事后回放。",
        "This analysis was performed after D4 labels became available and is an exploratory post-hoc replay only.",
        lang,
    ),
    danger=True,
)

metrics, error = load_csv(
    artifact("phase_d/d4_replay/d4_replay_metrics.csv"),
    ("analysis", "evaluation_label", "n_rows", "coverage", "macro_f1", "roc_auc_ovr"),
)
if error:
    empty_state(error)
else:
    all_rows = metrics.loc[metrics["analysis"] == "all"]
    selective = metrics.loc[metrics["analysis"] == "validation_frozen_selective"]
    if not all_rows.empty:
        row = all_rows.iloc[0]
        cards = st.columns(4)
        with cards[0]:
            metric_card("Macro F1", f"{row['macro_f1']:.3f}", "all replay rows")
        with cards[1]:
            metric_card("ROC-AUC OvR", f"{row['roc_auc_ovr']:.3f}", "all replay rows")
        with cards[2]:
            metric_card("Accuracy", f"{row['accuracy']:.3f}", "all replay rows")
        with cards[3]:
            metric_card("Rows", f"{int(row['n_rows'])}", "aggregate only")

    section_header(bilingual("完整回放与选择性回放", "Full and selective replay", lang))
    plot = metrics.copy()
    plot["Analysis"] = plot["analysis"].map({"all": "All", "validation_frozen_selective": "Frozen selective"})
    melted = plot.melt(id_vars=["Analysis", "coverage"], value_vars=["macro_f1", "balanced_accuracy", "roc_auc_ovr"], var_name="Metric", value_name="Score")
    melted["Metric"] = melted["Metric"].map({"macro_f1": "Macro F1", "balanced_accuracy": "Balanced accuracy", "roc_auc_ovr": "ROC-AUC OvR"})
    chart = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Analysis:N", title=None),
            xOffset="Metric:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1.0]), title="Score"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=[COLORS["blue"], COLORS["teal"], COLORS["amber"]]), legend=alt.Legend(orient="top")),
            tooltip=["Analysis:N", "Metric:N", alt.Tooltip("Score:Q", format=".3f"), alt.Tooltip("coverage:Q", format=".1%")],
        )
        .properties(height=310)
    )
    show_chart(chart)

    if not selective.empty:
        row = selective.iloc[0]
        interpretation_box(
            bilingual(
                f"验证集冻结的选择性规则在 D4 回放中保留 {row['coverage']:.1%}，Macro F1 为 {row['macro_f1']:.3f}。该结果不能升级证据等级。",
                f"The validation-frozen selective rule retains {row['coverage']:.1%} in D4 replay with Macro F1 {row['macro_f1']:.3f}. This does not upgrade the evidence level.",
                lang,
            )
        )

st.caption(EXPLORATORY_LABEL)
