from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.artifacts import artifact, load_csv
from dashboard.charts import show_chart
from dashboard.components import chart_heading, empty_state, interpretation_box, metric_card, page_header, section_header, status_badge
from dashboard.i18n import bilingual, get_language
from dashboard.theme import COLORS


lang = get_language()
page_header(
    bilingual("可信预测", "TRUSTWORTHY PREDICTION", lang),
    bilingual("校准与不确定性", "Calibration & Uncertainty", lang),
    bilingual("展示验证集选定校准器与置信度阈值在锁定时间测试中的行为。", "Validation-selected calibration and confidence thresholds evaluated on the locked temporal test.", lang),
)
status_badge(bilingual("验证集选择 · 测试集只评估", "Selected on validation · test evaluated once", lang), "success")

calibration, calibration_error = load_csv(
    artifact("phase_d/calibration/calibration_results.csv"),
    ("method", "split", "log_loss", "brier_score", "ece"),
)
test_cal = calibration.loc[calibration["split"] == "locked_temporal_test"].copy() if not calibration.empty else pd.DataFrame()
if calibration_error or test_cal.empty:
    empty_state(calibration_error or "Calibration artifacts are unavailable.")
else:
    uncal = test_cal.loc[test_cal["method"] == "uncalibrated"].iloc[0]
    iso = test_cal.loc[test_cal["method"] == "isotonic"].iloc[0]
    cards = st.columns(4)
    with cards[0]:
        metric_card("Log Loss", f"{iso['log_loss']:.3f}", f"from {uncal['log_loss']:.3f}")
    with cards[1]:
        metric_card("Brier Score", f"{iso['brier_score']:.3f}", f"from {uncal['brier_score']:.3f}")
    with cards[2]:
        metric_card("ECE", f"{iso['ece']:.3f}", f"from {uncal['ece']:.3f}")
    with cards[3]:
        metric_card(bilingual("选定方法", "Selected method", lang), "Isotonic", bilingual("仅由验证集选择", "Validation-selected", lang))

    section_header(bilingual("校准方法比较", "Calibration method comparison", lang))
    plot = test_cal.melt(id_vars="method", value_vars=["log_loss", "brier_score", "ece"], var_name="Metric", value_name="Score")
    metric_names = {"log_loss": "Log Loss", "brier_score": "Brier Score", "ece": "ECE"}
    plot["Metric"] = plot["Metric"].map(metric_names)
    chart_heading("Locked temporal test", bilingual("三项指标均为越低越好。", "Lower is better for all three metrics.", lang))
    bars = (
        alt.Chart(plot)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("method:N", title="Calibration method"),
            xOffset="Metric:N",
            y=alt.Y("Score:Q", title="Score"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=[COLORS["blue"], COLORS["teal"], COLORS["amber"]]), legend=alt.Legend(orient="top")),
            tooltip=["method:N", "Metric:N", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=300)
    )
    show_chart(bars)

selective, selective_error = load_csv(
    artifact("phase_d/uncertainty/selective_prediction.csv"),
    ("split", "target_coverage", "coverage", "threshold", "macro_f1", "error_rate"),
)
test_selective = selective.loc[selective["split"] == "locked_temporal_test"].copy() if not selective.empty else pd.DataFrame()
if selective_error or test_selective.empty:
    empty_state(selective_error or "Selective prediction artifact is unavailable.")
else:
    selected = test_selective.loc[test_selective["selected_validation_threshold"].astype(str).str.lower() == "true"]
    if selected.empty:
        selected = test_selective.iloc[[2]]
    point = selected.iloc[0]
    section_header(bilingual("风险–覆盖权衡", "Risk–coverage trade-off", lang))
    summary = st.columns(4)
    with summary[0]:
        metric_card(bilingual("实际覆盖率", "Observed coverage", lang), f"{point['coverage']:.1%}", "target 80%")
    with summary[1]:
        metric_card("Macro F1", f"{point['macro_f1']:.3f}", f"n={int(point['n_retained'])}")
    with summary[2]:
        metric_card(bilingual("错误率", "Error rate", lang), f"{point['error_rate']:.1%}", "locked temporal test")
    with summary[3]:
        metric_card(bilingual("置信阈值", "Confidence threshold", lang), f"{point['threshold']:.3f}", bilingual("验证集冻结", "Frozen on validation", lang))

    chart_frame = test_selective.sort_values("coverage")
    chart_heading(
        bilingual("保留覆盖率与错误率", "Retained coverage and error rate", lang),
        bilingual("曲线仅展示冻结阈值的测试集结果。", "The curve shows test behavior at frozen thresholds only.", lang),
    )
    line = (
        alt.Chart(chart_frame)
        .mark_line(point=True, strokeWidth=2.5, color=COLORS["teal"])
        .encode(
            x=alt.X("coverage:Q", title="Coverage", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.5, 1.02])),
            y=alt.Y("error_rate:Q", title="Error rate", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("coverage:Q", format=".1%"), alt.Tooltip("error_rate:Q", format=".1%"), alt.Tooltip("macro_f1:Q", format=".3f")],
        )
        .properties(height=310)
    )
    selected_point = alt.Chart(selected).mark_point(size=170, filled=True, color=COLORS["amber"]).encode(x="coverage:Q", y="error_rate:Q")
    show_chart(line + selected_point)

interpretation_box(
    bilingual(
        "等渗校准改善 Log Loss 和 ECE，但 Brier Score 基本不变；80% 目标覆盖阈值在测试集实际保留 81.1%，Macro F1 为 0.926。",
        "Isotonic calibration improves Log Loss and ECE while Brier Score is essentially unchanged; the 80% target threshold retains 81.1% with Macro F1 0.926.",
        lang,
    )
)
