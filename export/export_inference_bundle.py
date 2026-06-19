# ================================================================
# EXPORT CELL — Build deployment-ready inference bundle (SELF-CONTAINED)
# ================================================================
# Paste this as a NEW cell — works in a fresh Kaggle session, no need
# to re-run Part 2 training first. It defines the FT-Transformer
# classes itself (copied from nhanesv5.ipynb Cell 1), so joblib.load
# can reconstruct the pickled FTTransformer object.
#
# It does NOT retrain anything. It only:
#   1. Loads full_models / meta_lr / meta_sc / full_sc from
#      part2_state.joblib (mapping CUDA tensors -> CPU)
#   2. Recreates the exact 90/10 calibration split used during training
#      (same train_test_split call, same SEED -> identical indices)
#   3. Refits each base model's Platt calibrator on that split
#   4. Saves everything needed for single-patient inference to
#      /kaggle/working/nhanes_inference_bundle.joblib
#      /kaggle/working/ft_state_dict.pt
#
# Download both files and place them in models/ in this repo.
# ================================================================
import math
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ── Force CPU mapping for any embedded torch tensors ───────────
# (needed if this file was saved from a CUDA session, or if you're
# running this export cell on a CPU-only Kaggle session)
_orig_torch_load = torch.load
def _cpu_torch_load(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return _orig_torch_load(*args, **kwargs)
torch.load = _cpu_torch_load

# ── FT-Transformer class defs (copied from nhanesv5.ipynb Cell 1) ──
# MUST be defined before joblib.load(), since the pickled
# full_models["ft"] object needs these classes to reconstruct.
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_token, n_heads, attn_dropout=0.1):
        super().__init__()
        self.n_heads   = n_heads
        self.d_head    = d_token // n_heads
        self.scale     = self.d_head ** -0.5
        self.qkv       = nn.Linear(d_token, 3 * d_token, bias=False)
        self.proj      = nn.Linear(d_token, d_token)
        self.attn_drop = nn.Dropout(attn_dropout)
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop((q @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(B, N, D))

class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_factor=4, attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = MultiHeadSelfAttention(d_token, n_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_factor), nn.GELU(),
            nn.Dropout(ffn_dropout), nn.Linear(d_token * ffn_factor, d_token))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=192, n_heads=8, n_layers=3,
                 ffn_factor=4, attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_factor, attn_dropout, ffn_dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls    = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens[:, 0])).squeeze(-1)

# ================================================================
# Load checkpoint
# ================================================================
STATE_PATH = "/kaggle/input/datasets/mdalmahmudsiyam/deployjoblib/hpart2_state.joblib"
OUT_BUNDLE = "/kaggle/working/nhanes_inference_bundle.joblib"
OUT_FT_WEIGHTS = "/kaggle/working/ft_state_dict.pt"

print("Loading", STATE_PATH, "...")
state = joblib.load(STATE_PATH)

FEATS = state["FEATS"]
MODEL_KEYS = state["MODEL_KEYS"]
SEED = state["SEED"]
X_tv = state["X_tv"]
y_tv = state["y_tv"]
full_models = state["full_models"]
meta_lr = state["meta_lr"]
meta_sc = state["meta_sc"]
full_sc = state["full_sc"]
test_prob = state.get("test_prob", {})

print(f"FEATS ({len(FEATS)}): {FEATS}")
print(f"MODEL_KEYS: {MODEL_KEYS}")

# ── Recreate exact calibration split (same call as in run_stack_pipeline) ───
tv_tr, tv_cal = train_test_split(
    np.arange(len(X_tv)), test_size=0.10, stratify=y_tv, random_state=SEED
)
X_cal_raw = X_tv[tv_cal]
X_cal_scaled = full_sc.transform(X_cal_raw).astype(np.float32)
y_cal = y_tv[tv_cal]


def _get_ft_probs(model, X_np, batch_size=512, device="cpu"):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            b = torch.tensor(X_np[i:i + batch_size], dtype=torch.float32).to(device)
            probs.extend(torch.sigmoid(model(b)).cpu().numpy())
    return np.array(probs)


def _get_svm_probs(model, X):
    scores = model.decision_function(X)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


# ── Refit a Platt calibrator per base model on the calibration split ───────
calibrators = {}
for key in MODEL_KEYS:
    m = full_models[key]
    if key in ("lgbm", "xgb"):
        raw = m.predict_proba(X_cal_raw)[:, 1]
    elif key == "ft":
        m_cpu = m.to("cpu")
        raw = _get_ft_probs(m_cpu, X_cal_scaled)
    elif key == "svm":
        raw = _get_svm_probs(m, X_cal_scaled)
    else:  # rf, gbm
        raw = m.predict_proba(X_cal_scaled)[:, 1]

    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(raw.reshape(-1, 1), y_cal)
    calibrators[key] = lr
    print(f"  calibrator fit for '{key}' on {len(y_cal)} calibration rows")

# ── Training medians for every FEATS column ─────────────────────────────────
TRAIN_MEDIANS = {f: float(np.median(X_tv[:, i])) for i, f in enumerate(FEATS)}

# ── Risk tier cut-points + operating threshold ──────────────────────────────
risk_cutpoints = {"q33": 0.10, "q66": 0.25}
if "stack" in test_prob:
    q33, q66 = np.percentile(test_prob["stack"], [33, 66])
    risk_cutpoints = {"q33": float(q33), "q66": float(q66)}
    print(f"Risk cut-points from test_prob['stack']: q33={q33:.4f}  q66={q66:.4f}")
else:
    print("WARNING: 'test_prob' not found in state — using placeholder risk cut-points.")

operating_threshold = 0.205  # Stacking (Ours) row, TABLE_main_results_paper_ready.csv

# ── Save FT-Transformer weights separately ──────────────────────────────────
ft_model = full_models["ft"].to("cpu")
torch.save(ft_model.state_dict(), OUT_FT_WEIGHTS)
ft_config = {"n_features": len(FEATS), "d_token": 192, "n_heads": 8, "n_layers": 3}

sklearn_models = {k: v for k, v in full_models.items() if k != "ft"}

bundle = {
    "FEATS": FEATS,
    "MODEL_KEYS": MODEL_KEYS,
    "sklearn_models": sklearn_models,
    "ft_config": ft_config,
    "calibrators": calibrators,
    "meta_lr": meta_lr,
    "meta_sc": meta_sc,
    "full_sc": full_sc,
    "TRAIN_MEDIANS": TRAIN_MEDIANS,
    "risk_cutpoints": risk_cutpoints,
    "operating_threshold": operating_threshold,
}
joblib.dump(bundle, OUT_BUNDLE)

print("\n" + "=" * 60)
print("DONE")
print(f"  {OUT_BUNDLE}")
print(f"  {OUT_FT_WEIGHTS}")
print("Download both and place them in models/ in the GitHub repo.")
print("=" * 60)