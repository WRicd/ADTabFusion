from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.artifacts import artifact, load_csv
from dashboard.charts import show_chart
from dashboard.components import chart_heading, empty_state, interpretation_box, limitation_banner, metric_card, page_header, section_header, status_badge
from dashboard.i18n import bilingual, get_language
from dashboard.theme import COLORS


lang = get_language()
page_header(
    bilingual("临床风险层", "CLINICAL RISK LAYER", lang),
    bilingual("MCI 进展风险", "MCI Progression Risk", lang),
    bilingual("在预设 12、24、36、48 个月窗口内估计从 MCI 向 AD 的进展。", "Progression from MCI to AD over prespecified 12, 24, 36, and 48-month windows.", lang),
)
status_badge(bilingual("锁定时间测试", "Locked temporal test", lang), "success")

metrics, error = load_csv(
    artifact("phase_d/temporal_validation/mci_risk_metrics.csv"),
    ("horizon_months", "n_subjects", "roc_auc", "pr_auc", "macro_f1"),
)
if error:
    empty_state(error)
else:
    cards = st.columns(4)
    for column, (_, row) in zip(cards, metrics.sort_values("horizon_months").iterrows()):
        horizon = int(row["horizon_months"])
        with column:
            metric_card(f"{horizon}-month ROC-AUC", f"{row['roc_auc']:.3f}", f"n={int(row['n_subjects'])} · PR-AUC {row['pr_auc']:.3f}")

    section_header(bilingual("跨时间窗表现", "Performance by horizon", lang))
    plot = metrics.melt(
        id_vars=["horizon_months", "n_subjects"],
        value_vars=["roc_auc", "pr_auc", "macro_f1"],
        var_name="Metric",
        value_name="Score",
    )
    names = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "macro_f1": "Macro F1"}
    plot["Metric"] = plot["Metric"].map(names)
    chart_heading(
        bilingual("锁定测试表现", "Locked test performance", lang),
        bilingual("较长时间窗的样本量下降，48 个月结果需谨慎解读。", "Sample size falls at longer horizons; interpret the 48-month estimate cautiously.", lang),
    )
    chart = (
        alt.Chart(plot)
        .mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2.5)
        .encode(
            x=alt.X("horizon_months:O", title="Horizon (months)", sort=[12, 24, 36, 48]),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.4, 1.0]), title="Score"),
            color=alt.Color("Metric:N", scale=alt.Scale(domain=list(names.values()), range=[COLORS["blue"], COLORS["teal"], COLORS["amber"]]), legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("horizon_months:O", title="Months"), "Metric:N", alt.Tooltip("Score:Q", format=".3f"), "n_subjects:Q"],
        )
        .properties(height=330)
    )
    show_chart(chart)

cohort, cohort_error = load_csv(artifact("phase_d/cohorts/mci_landmark_summary.csv"))
if not cohort_error:
    section_header(bilingual("可评估队列", "Eligible cohorts", lang))
    display = cohort[["horizon_months", "eligible_subjects", "converters", "non_converters", "censored_subjects", "class_prevalence"]].copy()
    display.columns = ["Horizon (months)", "Eligible", "Converters", "Non-converters", "Censored", "Prevalence"]
    display["Prevalence"] = display["Prevalence"].map(lambda value: f"{value:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)

interpretation_box(
    bilingual(
        "24–36 个月窗口在锁定时间测试中达到约 0.80 ROC-AUC，适合作为队列级风险分层证据，而不是个体临床决策工具。",
        "The 24–36 month windows reach about 0.80 ROC-AUC on the locked temporal test and support cohort-level stratification, not individual clinical decisions.",
        lang,
    )
)
limitation_banner(
    bilingual(
        "48 个月测试仅包含 22 名受试者，属于小样本探索性结果。",
        "The 48-month test contains only 22 subjects and is a small-sample exploratory result.",
        lang,
    )
)
