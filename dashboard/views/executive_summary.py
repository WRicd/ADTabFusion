from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.artifacts import artifact, load_csv
from dashboard.charts import show_chart
from dashboard.components import (
    chart_heading,
    empty_state,
    interpretation_box,
    limitation_banner,
    metric_card,
    page_header,
    section_header,
    status_badge,
)
from dashboard.i18n import bilingual, get_language
from dashboard.theme import COLORS


lang = get_language()
page_header(
    bilingual("PHASE D 冻结证据", "PHASE D FROZEN EVIDENCE", lang),
    "AD-TabFusion",
    bilingual(
        "从纵向临床表格数据估计未来诊断状态、MCI 进展风险与可信预测范围。",
        "Future diagnosis, MCI progression risk, and trustworthy prediction coverage from longitudinal clinical tables.",
        lang,
    ),
)
status_badge(bilingual("只读 · 不重新训练", "Read-only · no retraining", lang), "success")

metrics, error = load_csv(
    artifact("phase_d/temporal_validation/transition_model_results.csv"),
    ("split", "macro_f1", "roc_auc_ovr", "balanced_accuracy"),
)
selected = metrics.loc[metrics["split"] == "locked_temporal_test"] if not metrics.empty else pd.DataFrame()

columns = st.columns(4)
if error or selected.empty:
    with columns[0]:
        empty_state(error or "Locked temporal result is unavailable.")
else:
    row = selected.iloc[0]
    with columns[0]:
        metric_card("Macro F1", f"{row['macro_f1']:.3f}", bilingual("锁定时间测试集", "Locked temporal test", lang))
    with columns[1]:
        metric_card("ROC-AUC OvR", f"{row['roc_auc_ovr']:.3f}", bilingual("三分类", "Three-class", lang))
    with columns[2]:
        metric_card("Balanced Accuracy", f"{row['balanced_accuracy']:.3f}", bilingual("锁定时间测试集", "Locked temporal test", lang))
    with columns[3]:
        metric_card("Subjects", f"{int(row['n_subjects'])}", f"{int(row['n_rows'])} prediction pairs")

section_header(
    bilingual("核心发现", "Core findings", lang),
    bilingual("结果均来自现有冻结 CSV，不在看板内计算或拟合。", "All values come from frozen CSV artifacts; the dashboard performs no fitting.", lang),
)

ablation, ablation_error = load_csv(
    artifact("phase_d/internal_validation/transition_ablation.csv"),
    ("ablation", "macro_f1"),
)
if ablation_error:
    empty_state(ablation_error)
else:
    labels = {
        "features_only": "Features only",
        "features_plus_forecast": "+ Forecast horizon",
        "features_plus_source_dx": "+ Source diagnosis",
        "features_plus_source_dx_forecast": "+ Source diagnosis + horizon",
    }
    plot = ablation.assign(display=ablation["ablation"].map(labels).fillna(ablation["ablation"]))
    chart_heading(
        bilingual("转归建模消融", "Transition modeling ablation", lang),
        bilingual("验证集 Macro F1；加入当前诊断状态带来主要增益。", "Validation Macro F1; source diagnosis provides the main gain.", lang),
    )
    chart = (
        alt.Chart(plot)
        .mark_bar(cornerRadiusEnd=3, color=COLORS["blue"], size=24)
        .encode(
            x=alt.X("macro_f1:Q", title="Macro F1", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("display:N", title=None, sort=list(labels.values())),
            tooltip=[alt.Tooltip("display:N", title="Ablation"), alt.Tooltip("macro_f1:Q", format=".3f")],
        )
        .properties(height=230)
    )
    show_chart(chart)

interpretation_box(
    bilingual(
        "模型在严格时间外推测试中保持 Macro F1 0.881；最明确的建模信号来自当前诊断状态与未来时间跨度的联合表示。",
        "The model retains Macro F1 0.881 under locked temporal testing; the clearest modeling signal comes from combining source diagnosis with forecast horizon.",
        lang,
    )
)

section_header(bilingual("研究流程", "Research flow", lang))
steps = [
    bilingual("TADPOLE 表格", "TADPOLE tables", lang),
    bilingual("泄漏审计", "Leakage audit", lang),
    bilingual("受试者级时间划分", "Subject-level temporal split", lang),
    bilingual("冻结模型与校准器", "Frozen models & calibrators", lang),
    bilingual("聚合证据展示", "Aggregate evidence display", lang),
]
st.markdown('<div class="ad-method-flow">' + "".join(f'<div class="ad-method-step">{step}</div>' for step in steps) + "</div>", unsafe_allow_html=True)

limitation_banner(
    bilingual(
        "D4 始终属于探索性事后回放，不是独立的验证性外部验证。",
        "D4 is always an exploratory post-hoc replay, not independent confirmatory external validation.",
        lang,
    )
)

cta = st.columns(3)
with cta[0]:
    st.page_link("views/transition_aware.py", label=bilingual("转归感知模型", "Transition-Aware Model", lang), use_container_width=True)
with cta[1]:
    st.page_link("views/mci_progression.py", label=bilingual("MCI 进展风险", "MCI Progression Risk", lang), use_container_width=True)
with cta[2]:
    st.page_link("views/calibration_uncertainty.py", label=bilingual("校准与不确定性", "Calibration & Uncertainty", lang), use_container_width=True)
