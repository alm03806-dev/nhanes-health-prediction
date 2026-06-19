---
title: NHANES Diabetes Risk Predictor
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
---

# NHANES Diabetes / Prediabetes Risk Predictor

A 6-model stacking ensemble (LightGBM, XGBoost, FT-Transformer, Random
Forest, Gradient Boosting, SVM + a logistic-regression meta-learner)
trained on NHANES 2011–2018 (cycles G–J) and validated on NHANES 2021–2023
(cycle L). ROC-AUC 0.895 on the held-out validation cycle.

Three tabs:
- **Risk Prediction** — enter a hypothetical patient's values, get a
  calibrated risk probability and tier
- **Model Performance** — ROC/PR curves, calibration, SHAP, subgroup
  robustness, DeLong tests — all from the actual validation run
- **Results Dashboard** — the underlying metric tables

Diabetes/prediabetes here is defined as HbA1c ≥ 6.5% or fasting glucose
≥ 126 mg/dL. This is a research / portfolio project, **not a medical
device** — it isn't intended for diagnosis or treatment decisions.

Full code, training notebooks, and methodology:
[github.com/alm03806-dev/nhanes-health-prediction](https://github.com/alm03806-dev/nhanes-health-prediction)
