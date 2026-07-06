# AD-TabFusion

A lightweight, leakage-aware ADNI/TADPOLE D1_D2 tabular learning demo for Alzheimer's disease diagnosis prediction.

This project prioritizes local reproducibility, subject-level splitting, interpretable sklearn baselines, modality ablation, missing-modality robustness, and a Streamlit dashboard for demonstration.

## Current Data Scope

This version uses only:

```text
data/raw/tadpole/TADPOLE_D1_D2.csv
```

Current active modalities:

- demographic: `AGE`, `PTGENDER`, `PTEDUCAT`
- cognitive: cognitive scores plus baseline-only `FAQ_bl`
- MRI-derived: regional volume features
- genetic: `APOE4`

Unavailable in the current CSV:

- PET: `FDG`, `AV45`, `PIB`
- CSF: `ABETA`, `TAU`, `PTAU`
- D3 external testing

## Install

```bash
pip install -r requirements.txt
```

## Run Pipeline

```bash
python scripts/check_leakage.py --config configs/tadpole_multiclass.yaml
python scripts/prepare_tadpole.py --config configs/tadpole_multiclass.yaml
python scripts/run_baselines.py --config configs/tadpole_baseline_only.yaml
python scripts/run_baselines.py --config configs/tadpole_all_visits.yaml
python scripts/run_modality_ablation.py --config configs/ablation_available_modalities.yaml
python scripts/run_missing_modality.py --config configs/ablation_available_modalities.yaml
python scripts/run_explainability.py --config configs/tadpole_all_visits.yaml
python scripts/analyze_error_cases.py --config configs/tadpole_all_visits.yaml
python scripts/generate_report.py --output outputs/reports/final_report.md
```

For a fast smoke run:

```bash
python scripts/run_baselines.py --config configs/tadpole_all_visits.yaml --quick
python scripts/run_modality_ablation.py --config configs/ablation_available_modalities.yaml --quick
python scripts/run_missing_modality.py --config configs/ablation_available_modalities.yaml --quick
python scripts/run_explainability.py --config configs/tadpole_all_visits.yaml --quick
```

## Dashboard

Start the dashboard from the project root:

```bash
streamlit run dashboard/app.py
```

If `streamlit` is not on your PATH, use:

```bash
python -m streamlit run dashboard/app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard reads existing files from `outputs/` and does not retrain models. Run the pipeline first if the dashboard shows missing tables or figures.

Use the sidebar `Language` selector to switch between English and Chinese dashboard text.

## Overview Figures

Generate overview figures:

```bash
python scripts/generate_overview_figures.py
```

Outputs are written to:

```text
outputs/figures/overview/
```

## Outputs

- `outputs/reports/leakage_check.md`
- `outputs/metrics/feature_blacklist.json`
- `outputs/metrics/baseline_results_by_seed.csv`
- `outputs/metrics/baseline_results_summary.csv`
- `outputs/metrics/modality_ablation_by_seed.csv`
- `outputs/metrics/modality_ablation_summary.csv`
- `outputs/metrics/missing_modality_results.csv`
- `outputs/metrics/feature_importance_best_model.csv`
- `outputs/metrics/modality_importance_best_model.csv`
- `outputs/reports/error_case_summary.md`
- `outputs/reports/final_report.md`

## Test

```bash
pytest tests/
```

---

# AD-TabFusion 中文说明

AD-TabFusion 是一个面向 ADNI/TADPOLE D1_D2 表格数据的轻量级、可复现、强调防止数据泄露的阿尔茨海默病诊断预测 demo。

当前版本重点是：

- 本地可复现运行
- subject-level 数据切分，避免同一受试者进入多个 split
- 泄露字段检查
- sklearn 强基线模型
- 模态消融实验
- 缺失模态鲁棒性分析
- 可解释性分析
- Streamlit 本地数据看板
- PPT 展示用总览图表

## 当前数据范围

当前版本只使用：

```text
data/raw/tadpole/TADPOLE_D1_D2.csv
```

当前可用模态：

- 人口学信息：`AGE`, `PTGENDER`, `PTEDUCAT`
- 认知量表：`MMSE`, `ADAS11`, `ADAS13`, `CDRSB`, RAVLT 系列指标，以及 baseline-only `FAQ_bl`
- MRI 派生体积特征：`Ventricles`, `Hippocampus`, `WholeBrain`, `Entorhinal`, `Fusiform`, `MidTemp`, `ICV`
- 遗传信息：`APOE4`

当前不可用：

- PET：`FDG`, `AV45`, `PIB`
- CSF：`ABETA`, `TAU`, `PTAU`
- D3 外部测试集

## 安装依赖

在项目根目录运行：

```bash
pip install -r requirements.txt
```

## 运行实验流水线

完整运行：

```bash
python scripts/check_leakage.py --config configs/tadpole_multiclass.yaml
python scripts/prepare_tadpole.py --config configs/tadpole_multiclass.yaml
python scripts/run_baselines.py --config configs/tadpole_baseline_only.yaml
python scripts/run_baselines.py --config configs/tadpole_all_visits.yaml
python scripts/run_modality_ablation.py --config configs/ablation_available_modalities.yaml
python scripts/run_missing_modality.py --config configs/ablation_available_modalities.yaml
python scripts/run_explainability.py --config configs/tadpole_all_visits.yaml
python scripts/analyze_error_cases.py --config configs/tadpole_all_visits.yaml
python scripts/generate_report.py --output outputs/reports/final_report.md
```

快速验证运行：

```bash
python scripts/run_baselines.py --config configs/tadpole_all_visits.yaml --quick
python scripts/run_modality_ablation.py --config configs/ablation_available_modalities.yaml --quick
python scripts/run_missing_modality.py --config configs/ablation_available_modalities.yaml --quick
python scripts/run_explainability.py --config configs/tadpole_all_visits.yaml --quick
```

## 启动数据看板

在项目根目录运行：

```bash
streamlit run dashboard/app.py
```

如果系统提示找不到 `streamlit` 命令，改用：

```bash
python -m streamlit run dashboard/app.py
```

启动后，终端会打印一个本地访问地址，通常是：

```text
http://localhost:8501
```

在浏览器中打开该地址即可查看数据看板。

看板只读取 `outputs/` 目录下已有结果，不会重新训练模型。如果页面显示某些表格或图不存在，请先运行上面的实验流水线。

看板左侧提供 `语言` 选择框，可在中文和英文界面之间切换。

## 生成 PPT 总览图

```bash
python scripts/generate_overview_figures.py
```

输出目录：

```text
outputs/figures/overview/
```

其中包含：

- `pipeline_overview.png`
- `dataset_modalities_overview.png`
- `model_performance_overview.png`
- `modality_ablation_overview.png`
- `missing_modality_overview.png`
- `explainability_overview.png`
- `README.md`

## 主要输出文件

- `outputs/reports/leakage_check.md`
- `outputs/metrics/feature_blacklist.json`
- `outputs/metrics/baseline_results_by_seed.csv`
- `outputs/metrics/baseline_results_summary.csv`
- `outputs/metrics/modality_ablation_by_seed.csv`
- `outputs/metrics/modality_ablation_summary.csv`
- `outputs/metrics/missing_modality_results.csv`
- `outputs/metrics/feature_importance_best_model.csv`
- `outputs/metrics/modality_importance_best_model.csv`
- `outputs/reports/error_case_summary.md`
- `outputs/reports/final_report.md`

## 测试

```bash
pytest tests/
```
