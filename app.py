"""
NHANES Diabetes / Prediabetes Risk — Gradio app for Hugging Face Spaces.

Tabs:
  1. Risk Prediction   — live stacking-ensemble inference for one patient
  2. Model Performance  — real evaluation figures from the paper run
  3. Results Dashboard   — real metric tables from the paper run

Run locally:
    pip install -r requirements.txt
    python app.py
"""
import os
import pandas as pd
import gradio as gr

from src.ui_constants import FEATURE_GROUPS
from src.inference import get_model

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
FIGURES_DIR = os.path.join(ASSETS_DIR, "figures")
TABLES_DIR = os.path.join(ASSETS_DIR, "tables")

CSS = """
:root {
    --nh-bg: #F6F9F9;
    --nh-card: #FFFFFF;
    --nh-ink: #1E2A2B;
    --nh-muted: #5C6B6C;
    --nh-teal: #0B6E6E;
    --nh-teal-dark: #08504F;
    --nh-low: #1B8A5A;
    --nh-medium: #C7821A;
    --nh-high: #C0392B;
    --nh-border: #DCE6E6;
}
.gradio-container { background: var(--nh-bg) !important; font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
#nh-header {
    background: linear-gradient(135deg, var(--nh-teal) 0%, var(--nh-teal-dark) 100%);
    color: white; padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;
}
#nh-header h1 { margin: 0 0 4px 0; font-size: 1.5rem; font-weight: 700; }
#nh-header p { margin: 0; opacity: 0.9; font-size: 0.92rem; }
.nh-result-card {
    border: 1px solid var(--nh-border); border-radius: 12px; padding: 18px 20px;
    background: var(--nh-card);
}
.nh-tier-low   { color: var(--nh-low);    font-weight: 700; }
.nh-tier-medium{ color: var(--nh-medium); font-weight: 700; }
.nh-tier-high  { color: var(--nh-high);   font-weight: 700; }
.nh-disclaimer {
    font-size: 0.8rem; color: var(--nh-muted); border-top: 1px solid var(--nh-border);
    margin-top: 14px; padding-top: 10px;
}
"""

RISK_TIER_CLASS = {"Low": "nh-tier-low", "Medium": "nh-tier-medium", "High": "nh-tier-high"}


def build_input_components():
    components = {}
    blocks = []
    CHUNK = 3
    for group_name, fields in FEATURE_GROUPS:
        with gr.Accordion(group_name, open=(group_name == "Demographics")) as acc:
            for i in range(0, len(fields), CHUNK):
                with gr.Row():
                    for feat, label, kind, default, extra in fields[i:i + CHUNK]:
                        if kind == "slider":
                            lo, hi, step = extra
                            comp = gr.Slider(minimum=lo, maximum=hi, step=step, value=default, label=label)
                        else:  # dropdown
                            comp = gr.Dropdown(choices=list(extra.keys()), value=default, label=label)
                        components[feat] = comp
        blocks.append(acc)
    return components, blocks


def run_prediction(*values):
    model = get_model()
    feat_names = list(_INPUT_ORDER)
    raw = dict(zip(feat_names, values))

    # Map dropdown display labels back to NHANES codes
    decoded = {}
    for group_name, fields in FEATURE_GROUPS:
        for feat, label, kind, default, extra in fields:
            v = raw[feat]
            decoded[feat] = extra[v] if kind == "dropdown" else float(v)

    if not model.loaded:
        msg = (
            "### Model not loaded yet\n\n"
            f"{model.error}\n\n"
            "The form above is fully functional — once `models/nhanes_inference_bundle.joblib` "
            "and `models/ft_state_dict.pt` are added to this Space, predictions will work "
            "without any other changes."
        )
        return msg, gr.update(visible=False)

    result = model.predict(decoded)
    prob_pct = result["probability"] * 100
    tier = result["risk_tier"]
    tier_class = RISK_TIER_CLASS[tier]

    breakdown_rows = [
        {"Base model": k.upper(), "Calibrated probability": f"{v * 100:.1f}%"}
        for k, v in result["per_model_calibrated"].items()
    ]
    breakdown_df = pd.DataFrame(breakdown_rows)

    headline = f"""
<div class="nh-result-card">
<h3 style="margin-top:0;">Estimated risk: {prob_pct:.1f}%</h3>
<p>Risk tier: <span class="{tier_class}">{tier}</span>
(population tertile, based on the held-out NHANES test set)</p>
<p style="color:var(--nh-muted); font-size:0.85rem;">
Stacking ensemble of LightGBM, XGBoost, FT-Transformer, Random Forest, Gradient Boosting and SVM,
combined by a logistic-regression meta-learner. Each base model's probability is Platt-calibrated
before stacking.
</p>
<div class="nh-disclaimer">
This tool is a research / portfolio demonstration trained on NHANES survey data (cycles G–J,
validated on cycle L). It is <strong>not a medical device</strong> and must not be used for
diagnosis or treatment decisions. Diabetes/prediabetes status here is defined as
HbA1c &ge; 6.5% or fasting glucose &ge; 126 mg/dL.
</div>
</div>
"""
    return headline, gr.update(value=breakdown_df, visible=True)


def load_table(filename):
    path = os.path.join(TABLES_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame({"error": [f"{filename} not found in assets/tables/"]})
    return pd.read_csv(path)


def build_performance_tab():
    figures = [
        ("Fig2_ROC_curves.png", "ROC curves — all base models vs. the stacking ensemble"),
        ("Fig4_PR_curves.png", "Precision-Recall curves (imbalanced-class view)"),
        ("Fig7_calibration_analysis.png", "Calibration analysis (Brier score, ECE)"),
        ("Fig12_performance_bar.png", "Headline metric comparison across models"),
        ("Fig11_confusion_matrices.png", "Confusion matrices at the operating threshold"),
        ("Fig9_SHAP_LGBM.png", "SHAP feature importance (LightGBM base learner)"),
        ("Fig5_risk_stratification.png", "Risk stratification — Low / Medium / High tiers"),
        ("Fig6_subgroup_robustness.png", "Subgroup robustness (age, sex, race/ethnicity)"),
        ("Fig15_DeLong_comparison.png", "DeLong test — pairwise ROC-AUC comparisons"),
        ("Fig17_feature_group_ablation.png", "Feature-group ablation study"),
    ]
    gr.Markdown(
        "All figures below are generated directly from the validation run on NHANES "
        "cycle L (held-out, 2021–2023) — nothing here is illustrative or placeholder."
    )
    for i in range(0, len(figures), 2):
        with gr.Row():
            for fname, caption in figures[i:i + 2]:
                path = os.path.join(FIGURES_DIR, fname)
                if os.path.exists(path):
                    gr.Image(value=path, label=caption, show_label=True, interactive=False)


def build_dashboard_tab():
    gr.Markdown("### Main results — Stacking ensemble vs. baselines")
    gr.Dataframe(value=load_table("TABLE_main_results_paper_ready.csv"), wrap=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Risk stratification (test set)")
            gr.Dataframe(value=load_table("TABLE_risk_stratification.csv"))
        with gr.Column():
            gr.Markdown("### Calibration summary")
            gr.Dataframe(value=load_table("TABLE_calibration_summary.csv"))

    gr.Markdown("### Subgroup analysis (age / sex / race-ethnicity)")
    gr.Dataframe(value=load_table("TABLE_subgroup_analysis.csv"), wrap=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### DeLong pairwise ROC-AUC tests")
            gr.Dataframe(value=load_table("TABLE_delong_tests.csv"), wrap=True)
        with gr.Column():
            gr.Markdown("### Top 15 features by mean |SHAP|")
            gr.Dataframe(value=load_table("TABLE_top15_features.csv"))

    gr.Markdown("### Dataset summary")
    gr.Dataframe(value=load_table("dataset_summary.csv"))


with gr.Blocks(title="NHANES Diabetes Risk Predictor") as demo:
    gr.HTML(
        """
        <div id="nh-header">
            <h1>NHANES Diabetes / Prediabetes Risk Predictor</h1>
            <p>6-model stacking ensemble · trained on NHANES 2011-2018 (G-J) · validated on NHANES 2021-2023 (L)</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("🔬 Risk Prediction"):
            gr.Markdown(
                "Enter values for a hypothetical patient. Sliders default to roughly "
                "population-typical values — adjust whatever you'd like to test."
            )
            input_components, _ = build_input_components()
            _INPUT_ORDER = list(input_components.keys())

            predict_btn = gr.Button("Predict risk", variant="primary")
            result_html = gr.Markdown()
            breakdown_table = gr.Dataframe(label="Per-model calibrated probability", visible=False)

            predict_btn.click(
                fn=run_prediction,
                inputs=[input_components[f] for f in _INPUT_ORDER],
                outputs=[result_html, breakdown_table],
            )

        with gr.Tab("📊 Model Performance"):
            build_performance_tab()

        with gr.Tab("📋 Results Dashboard"):
            build_dashboard_tab()

    gr.Markdown(
        "---\n"
        "Built by [alm03806-dev](https://github.com/alm03806-dev) · "
        "[GitHub repo](https://github.com/alm03806-dev/nhanes-health-prediction) · "
        "Research / portfolio project — not a medical device."
    )

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="teal"))
