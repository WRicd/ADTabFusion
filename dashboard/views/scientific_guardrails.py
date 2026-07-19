from __future__ import annotations

import streamlit as st

from dashboard.components import interpretation_box, limitation_banner, page_header, section_header, status_badge
from dashboard.i18n import bilingual, get_language


lang = get_language()
page_header(
    bilingual("研究设计", "STUDY DESIGN", lang),
    bilingual("科研护栏", "Scientific Guardrails", lang),
    bilingual("明确区分训练、选择、锁定测试与探索性回放，避免展示层改变实验结论。", "Separating training, selection, locked testing, and exploratory replay so presentation cannot alter the experiment.", lang),
)
status_badge(bilingual("冻结 artifact 为唯一结果来源", "Frozen artifacts are the sole result source", lang), "success")

section_header(bilingual("防泄漏设计", "Leakage controls", lang))
controls = [
    (bilingual("受试者级隔离", "Subject-level isolation", lang), bilingual("同一受试者不会跨训练、验证和锁定时间测试集。", "A subject cannot cross training, validation, and locked temporal test splits.", lang)),
    (bilingual("只用历史特征", "Historical features only", lang), bilingual("预测时点之后的诊断、结局和派生目标不进入特征。", "Post-index diagnoses, outcomes, and target-derived fields do not enter features.", lang)),
    (bilingual("验证集选择", "Validation-only selection", lang), bilingual("模型、校准方法和选择性预测阈值只由验证集确定。", "Model, calibrator, and selective threshold choices are made on validation data only.", lang)),
    (bilingual("锁定测试", "Locked testing", lang), bilingual("时间测试集用于一次性冻结评估，不参与再拟合。", "The temporal test is used for frozen evaluation and never refitting.", lang)),
]
for title, body in controls:
    st.markdown(f"**{title}**")
    st.write(body)

section_header(bilingual("证据等级", "Evidence labels", lang))
evidence = st.columns(3)
with evidence[0]:
    st.markdown("**Internal validation**")
    st.write(bilingual("用于消融和模型选择。", "Used for ablation and model selection.", lang))
with evidence[1]:
    st.markdown("**Locked temporal test**")
    st.write(bilingual("主要内部外推证据。", "Primary internal extrapolation evidence.", lang))
with evidence[2]:
    st.markdown("**Exploratory replay**")
    st.write(bilingual("仅作事后稳健性观察。", "Post-hoc robustness observation only.", lang))

interpretation_box(
    bilingual(
        "Dashboard 本身不导入训练模块，不调用 fit、calibrate 或 dump，也不写入 outputs。",
        "The dashboard imports no training module, calls no fit, calibrate, or dump routine, and writes nothing to outputs.",
        lang,
    )
)
limitation_banner(
    bilingual(
        "D4 必须始终标记为 exploratory post-hoc replay，不能表述为独立的验证性外部验证。",
        "D4 must always be labeled exploratory post-hoc replay and cannot be described as independent confirmatory external validation.",
        lang,
    ),
    danger=True,
)

