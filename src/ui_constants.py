"""
Clinical feature definitions for the live demo.

These are the 57 real, clinically meaningful NHANES features the model
was trained on (FEATURE_COLS from nhanes_dataset_preparation_v3.ipynb,
minus the 3 leakage variables excluded in nhanesv5.ipynb: LBXSCR,
LBXSUA, LBXSAL).

IMPORTANT — read docs/DEPLOYMENT.md:
The trained model's actual FEATS list (saved inside
nhanes_inference_bundle.joblib) may contain a handful of extra
non-clinical columns that ended up in the training matrix as a
side-effect of the data-prep pipeline (SEQN, WTMEC4YR, WEIGHT_NORM,
SDMVPSU, SDMVSTRA — NHANES survey-design/ID columns, not patient
measurements). Those are NOT collected from the user here; src/inference.py
fills them with their training-set median value automatically, using
whatever the real FEATS list says at load time. Nothing about that
behavior needs to change even if the exact count of those extra
columns turns out to be different than expected — it is driven by the
bundle, not hardcoded.
"""

# ── Categorical code maps (standard NHANES codings) ─────────────────────────
SEX_MAP = {"Male": 1, "Female": 2}

RACE_MAP = {
    "Mexican American": 1,
    "Other Hispanic": 2,
    "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4,
    "Non-Hispanic Asian": 6,
    "Other / Multi-racial": 7,
}

EDUCATION_MAP = {
    "Less than 9th grade": 1,
    "9th-11th grade": 2,
    "High school grad / GED": 3,
    "Some college / AA degree": 4,
    "College graduate or above": 5,
}

MARITAL_MAP = {
    "Married / living with partner": 1,
    "Widowed / divorced / separated": 2,
    "Never married": 3,
}

YES_NO_MAP = {"Yes": 1, "No": 2}

# ── Grouped clinical inputs: (feature, label, kind, default, extra) ─────────
# kind: "slider" -> (min, max, step) in `extra`; "dropdown" -> map in `extra`
DEMOGRAPHICS = [
    ("RIDAGEYR", "Age (years)", "slider", 45, (18, 85, 1)),
    ("RIAGENDR", "Sex", "dropdown", "Male", SEX_MAP),
    ("RIDRETH3", "Race / ethnicity", "dropdown", "Non-Hispanic White", RACE_MAP),
    ("DMDEDUC2", "Education level", "dropdown", "High school grad / GED", EDUCATION_MAP),
    ("DMDMARTZ", "Marital status", "dropdown", "Married / living with partner", MARITAL_MAP),
    ("INDFMPIR", "Family income-to-poverty ratio (0-5)", "slider", 2.0, (0.0, 5.0, 0.05)),
]

ANTHROPOMETRY = [
    ("BMXWT", "Weight (kg)", "slider", 80.0, (30.0, 200.0, 0.5)),
    ("BMXHT", "Height (cm)", "slider", 170.0, (130.0, 210.0, 0.5)),
    ("BMXBMI", "BMI (kg/m^2)", "slider", 27.7, (12.0, 70.0, 0.1)),
    ("BMXWAIST", "Waist circumference (cm)", "slider", 95.0, (50.0, 175.0, 0.5)),
    ("BMXHIP", "Hip circumference (cm)", "slider", 102.0, (50.0, 175.0, 0.5)),
]

BLOOD_PRESSURE = [
    ("SBP_MEAN", "Systolic BP, mean (mmHg)", "slider", 120.0, (80.0, 220.0, 1.0)),
    ("DBP_MEAN", "Diastolic BP, mean (mmHg)", "slider", 78.0, (40.0, 130.0, 1.0)),
    ("BPXPLS", "Pulse rate (bpm)", "slider", 72.0, (40.0, 130.0, 1.0)),
]

LIPIDS = [
    ("LBXTC", "Total cholesterol (mg/dL)", "slider", 190.0, (100.0, 400.0, 1.0)),
    ("HDL", "HDL cholesterol (mg/dL)", "slider", 50.0, (15.0, 110.0, 1.0)),
    ("LBXTR", "Triglycerides (mg/dL)", "slider", 120.0, (30.0, 600.0, 1.0)),
    ("LBDLDL", "LDL cholesterol (mg/dL)", "slider", 110.0, (30.0, 300.0, 1.0)),
]

LABS = [
    ("LBXSATSI", "ALT (U/L)", "slider", 22.0, (5.0, 200.0, 1.0)),
    ("LBXSASSI", "AST (U/L)", "slider", 24.0, (5.0, 300.0, 1.0)),
    ("LBXSGTSI", "GGT (U/L)", "slider", 25.0, (5.0, 500.0, 1.0)),
    ("LBXSCLSI", "Chloride (mmol/L)", "slider", 103.0, (80.0, 120.0, 1.0)),
    ("LBXSNASI", "Sodium (mmol/L)", "slider", 140.0, (120.0, 160.0, 1.0)),
    ("LBXSKSI", "Potassium (mmol/L)", "slider", 4.2, (2.0, 7.0, 0.1)),
]

CBC = [
    ("LBXWBCSI", "White blood cells (1000 cells/uL)", "slider", 7.0, (2.0, 20.0, 0.1)),
    ("LBXRBCSI", "Red blood cells (million cells/uL)", "slider", 4.8, (2.0, 7.0, 0.1)),
    ("LBXHGB", "Hemoglobin (g/dL)", "slider", 14.5, (5.0, 20.0, 0.1)),
    ("LBXHCT", "Hematocrit (%)", "slider", 43.0, (15.0, 60.0, 0.5)),
    ("LBXPLTSI", "Platelets (1000 cells/uL)", "slider", 250.0, (50.0, 600.0, 5.0)),
    ("LBXNENO", "Segmented neutrophils (1000 cells/uL)", "slider", 4.0, (0.5, 15.0, 0.1)),
    ("LBXLYPCT", "Lymphocyte percent (%)", "slider", 30.0, (5.0, 65.0, 0.5)),
    ("LBXRDW", "Red cell distribution width (%)", "slider", 13.0, (10.0, 22.0, 0.1)),
]

LIFESTYLE = [
    ("SMQ020", "Smoked >=100 cigarettes in life", "dropdown", "No", YES_NO_MAP),
    ("ALQ101", "Had >=12 alcoholic drinks in any year", "dropdown", "No", YES_NO_MAP),
    ("PAQ605", "Does vigorous-intensity work activity", "dropdown", "No", YES_NO_MAP),
    ("SLD010H", "Usual sleep hours per night", "slider", 7.0, (2.0, 14.0, 0.5)),
]

MEDICAL_HISTORY = [
    ("MCQ160B", "Told had congestive heart failure", "dropdown", "No", YES_NO_MAP),
    ("MCQ160C", "Told had coronary heart disease", "dropdown", "No", YES_NO_MAP),
    ("MCQ160D", "Told had angina / angina pectoris", "dropdown", "No", YES_NO_MAP),
    ("MCQ160F", "Told had a stroke", "dropdown", "No", YES_NO_MAP),
]

DIETARY = [
    ("DIET_TKCAL", "Energy intake (kcal/day)", "slider", 2000.0, (500.0, 5000.0, 50.0)),
    ("DIET_TPROT", "Protein intake (g/day)", "slider", 80.0, (10.0, 300.0, 1.0)),
    ("DIET_TCARB", "Carbohydrate intake (g/day)", "slider", 250.0, (20.0, 600.0, 5.0)),
    ("DIET_TTFAT", "Total fat intake (g/day)", "slider", 75.0, (10.0, 250.0, 1.0)),
    ("DIET_TSUGR", "Total sugar intake (g/day)", "slider", 90.0, (0.0, 400.0, 1.0)),
    ("DIET_TFIBE", "Dietary fiber intake (g/day)", "slider", 16.0, (0.0, 80.0, 0.5)),
    ("DIET_TSODI", "Sodium intake (mg/day)", "slider", 3300.0, (200.0, 8000.0, 50.0)),
    ("DIET_TALCO", "Alcohol intake (g/day)", "slider", 5.0, (0.0, 150.0, 1.0)),
]

FEATURE_GROUPS = [
    ("Demographics", DEMOGRAPHICS),
    ("Anthropometry", ANTHROPOMETRY),
    ("Blood pressure", BLOOD_PRESSURE),
    ("Lipid panel", LIPIDS),
    ("Liver / kidney / electrolytes", LABS),
    ("Complete blood count", CBC),
    ("Lifestyle", LIFESTYLE),
    ("Medical history", MEDICAL_HISTORY),
    ("Dietary (24h recall avg.)", DIETARY),
]

# Raw NHANES/engineered feature names this UI collects or derives directly.
# Anything in the model's real FEATS list that is NOT in this set gets
# auto-filled from TRAIN_MEDIANS by src/inference.py.
ALL_UI_FEATURES = [f for _, grp in FEATURE_GROUPS for f, *_ in grp]

ENGINEERED_FROM_UI = [
    "PULSE_PRESSURE", "MAP", "WAIST_BMI_RATIO", "WAIST_HIP_RATIO",
    "OBESE", "ABD_OBESITY", "HYPERTENSION", "TC_HDL_RATIO", "TG_HDL_RATIO",
]

CLINICAL_FEATS = ALL_UI_FEATURES + ENGINEERED_FROM_UI


def engineer_features(raw: dict) -> dict:
    """Reproduce the exact feature-engineering steps from
    nhanes_dataset_preparation_v3.ipynb (Cell 5 / Cell 9) for a single
    patient's raw inputs."""
    out = dict(raw)
    sbp, dbp = raw["SBP_MEAN"], raw["DBP_MEAN"]
    out["PULSE_PRESSURE"] = sbp - dbp
    out["MAP"] = dbp + out["PULSE_PRESSURE"] / 3
    out["HYPERTENSION"] = int(sbp >= 130 or dbp >= 80)

    waist, bmi, hip = raw["BMXWAIST"], raw["BMXBMI"], raw["BMXHIP"]
    out["WAIST_BMI_RATIO"] = waist / (bmi + 1e-9)
    out["WAIST_HIP_RATIO"] = waist / (hip + 1e-9)
    out["OBESE"] = int(bmi >= 30)

    sex = raw["RIAGENDR"]
    out["ABD_OBESITY"] = int(waist > 102) if sex == 1 else int(waist > 88)

    tc, hdl, tr = raw["LBXTC"], raw["HDL"], raw["LBXTR"]
    out["TC_HDL_RATIO"] = tc / (hdl + 1e-9)
    out["TG_HDL_RATIO"] = tr / (hdl + 1e-9)

    return out
