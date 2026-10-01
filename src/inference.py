"""
Inference engine for the NHANES diabetes stacking ensemble.

Reproduces — for a single new patient — exactly what
nhanesv5.ipynb (Part 2, "full retraining for test predictions") does:

  1. LightGBM / XGBoost predict on raw (imputed, unscaled) features
  2. Random Forest / GBM / SVM / FT-Transformer predict on
     StandardScaler-scaled features (full_sc)
  3. Each base model's raw probability is passed through its own
     fitted Platt calibrator (a 1-feature LogisticRegression)
  4. The 6 calibrated probabilities are stacked, scaled with meta_sc,
     and passed to meta_lr for the final ensemble probability

This file expects a bundle produced by export/export_inference_bundle.py
(see docs/DEPLOYMENT.md). It does NOT retrain anything.
"""
import os
import joblib
import numpy as np
import torch

from .ft_transformer import FTTransformer, get_ft_probs, get_svm_probs
from .ui_constants import CLINICAL_FEATS, engineer_features

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "nhanes_inference_bundle.joblib")
FT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ft_state_dict.pt")


class NHANESModel:
    def __init__(self, bundle_path=BUNDLE_PATH, ft_weights_path=FT_WEIGHTS_PATH):
        self.loaded = False
        self.error = None
        if not (os.path.exists(bundle_path) and os.path.exists(ft_weights_path)):
            self.error = (
                f"Model files not found. Expected:\n  {os.path.abspath(bundle_path)}\n"
                f"  {os.path.abspath(ft_weights_path)}\n"
                "Run export/export_inference_bundle.py on Kaggle and copy both files "
                "into models/ before deploying. See docs/DEPLOYMENT.md."
            )
            return

        bundle = joblib.load(bundle_path)
        self.FEATS = bundle["FEATS"]
        self.MODEL_KEYS = bundle["MODEL_KEYS"]
        self.sklearn_models = bundle["sklearn_models"]
        self.calibrators = bundle["calibrators"]
        self.meta_lr = bundle["meta_lr"]
        self.meta_sc = bundle["meta_sc"]
        self.full_sc = bundle["full_sc"]
        self.train_medians = bundle["TRAIN_MEDIANS"]
        self.risk_cutpoints = bundle.get("risk_cutpoints", {"q33": 0.10, "q66": 0.25})
        self.operating_threshold = bundle.get("operating_threshold", 0.29)

        ft_config = bundle["ft_config"]
        self.ft_model = FTTransformer(
            n_features=ft_config["n_features"],
            d_token=ft_config["d_token"],
            n_heads=ft_config["n_heads"],
            n_layers=ft_config["n_layers"],
        )
        self.ft_model.load_state_dict(torch.load(ft_weights_path, map_location="cpu"))
        self.ft_model.eval()

        # Sanity check: every clinical UI feature this app collects should
        # actually be one of the model's FEATS. Anything in FEATS that the
        # UI doesn't collect gets auto-filled from training medians.
        self.extra_feats = [f for f in self.FEATS if f not in CLINICAL_FEATS]

        self.loaded = True

    def _build_feature_vector(self, raw_inputs: dict) -> np.ndarray:
        engineered = engineer_features(raw_inputs)
        row = []
        for f in self.FEATS:
            if f in engineered:
                row.append(float(engineered[f]))
            else:
                # Non-clinical artifact column (e.g. SEQN, survey design
                # columns) the UI does not collect — hold at its training
                # median so it cannot swing the prediction.
                row.append(float(self.train_medians.get(f, 0.0)))
        return np.array(row, dtype=np.float32).reshape(1, -1)

    def predict(self, raw_inputs: dict) -> dict:
        if not self.loaded:
            raise RuntimeError(self.error)

        X = self._build_feature_vector(raw_inputs)
        X_scaled = self.full_sc.transform(X).astype(np.float32)

        raw_probs = {}
        for key in self.MODEL_KEYS:
            if key == "ft":
                raw_probs[key] = get_ft_probs(self.ft_model, X_scaled)[0]
            elif key == "svm":
                raw_probs[key] = get_svm_probs(self.sklearn_models[key], X_scaled)[0]
            elif key in ("lgbm", "xgb"):
                raw_probs[key] = self.sklearn_models[key].predict_proba(X)[:, 1][0]
            else:  # rf, gbm
                raw_probs[key] = self.sklearn_models[key].predict_proba(X_scaled)[:, 1][0]

        calibrated = {}
        for key in self.MODEL_KEYS:
            cal_model = self.calibrators[key]
            p = np.array([[raw_probs[key]]])
            calibrated[key] = float(cal_model.predict_proba(p)[:, 1][0])

        meta_x = np.array([[calibrated[k] for k in self.MODEL_KEYS]])
        meta_x_scaled = self.meta_sc.transform(meta_x)
        stack_prob = float(self.meta_lr.predict_proba(meta_x_scaled)[:, 1][0])

        q33, q66 = self.risk_cutpoints["q33"], self.risk_cutpoints["q66"]
        if stack_prob < q33:
            tier = "Low"
        elif stack_prob < q66:
            tier = "Medium"
        else:
            tier = "High"

        return {
            "probability": stack_prob,
            "risk_tier": tier,
            "above_operating_threshold": stack_prob >= self.operating_threshold,
            "per_model_calibrated": calibrated,
            "per_model_raw": {k: float(v) for k, v in raw_probs.items()},
        }


_model_singleton = None


def get_model() -> NHANESModel:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = NHANESModel()
    return _model_singleton
