# Deployment guide

Three steps: generate the model bundle on Kaggle, push this repo to GitHub,
deploy it to Hugging Face Spaces.

## 1. Generate the inference bundle (Kaggle)

The CSV/PNG outputs in `assets/` are already real and already in this repo —
that part is done. What's missing is the trained model itself, because the
zip of outputs only contained result tables and figures, not the `.joblib`
checkpoints.

1. Open the Kaggle session/notebook that produced `part2_state.joblib`
   (Notebook 1 / Part 2 of `nhanesv5.ipynb`).
2. Make sure `/kaggle/working/part2_state.joblib` exists (re-run Part 2 if
   you've started a fresh session and it isn't there).
3. Paste the contents of `export/export_inference_bundle.py` as a new cell
   at the end of that notebook and run it. It does **not** retrain
   anything — it only reloads the already-fitted models and saves a
   deployment-ready bundle.
4. It produces two files in `/kaggle/working/`:
   - `nhanes_inference_bundle.joblib`
   - `ft_state_dict.pt`
5. Download both and place them in `models/` in this repo (overwriting
   nothing — that folder is otherwise empty).
6. Sanity check locally before deploying:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
   Open the local URL, go to "Risk Prediction", leave the defaults, click
   Predict. You should get a probability + risk tier, not the
   "Model not loaded" message.

**Worth knowing:** `export/export_inference_bundle.py` prints the exact
`FEATS` list it finds inside `part2_state.joblib`. If that list is longer
than the 57 clinical fields the UI collects (see the docstring at the top
of `src/ui_constants.py`), it means a few non-clinical columns (likely
`SEQN`, `SDMVPSU`, `SDMVSTRA`, `WTMEC4YR`, `WEIGHT_NORM`) ended up in the
trained feature set as a side effect of the data-prep pipeline. The app
handles this automatically — those columns get filled with their training
median rather than asked from the user — but it's worth knowing about for
the paper write-up, since reviewers may ask why survey-design columns are
in the feature importances (`WEIGHT_NORM` showing up at SHAP rank 8 in
`assets/tables/TABLE_top15_features.csv` is the visible symptom of this).

## 2. Push to GitHub

```bash
cd nhanes-health-prediction
git init
git add .
git commit -m "NHANES diabetes risk: stacking ensemble + Gradio demo"
git branch -M main
git remote add origin https://github.com/alm03806-dev/nhanes-health-prediction.git
git push -u origin main
```

If `models/*.joblib` / `models/*.pt` are over GitHub's 100 MB file limit,
either set up [Git LFS](https://git-lfs.com/) for them, or skip pushing
them to GitHub entirely and upload them straight to the HF Space instead
(Spaces doesn't need GitHub to have a copy) — that's simplest, and is what
the `.gitignore` in this repo defaults to.

## 3. Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Gradio**
   - Hardware: CPU basic (free tier) is enough for this model
2. Either:
   - **Link to GitHub**: in the Space settings, connect the GitHub repo
     so it auto-syncs, or
   - **Upload directly**: `git clone` the empty Space repo HF gives you,
     copy this project's files in, commit, push.
3. Rename `README_HF.md` (in this repo) to `README.md` *inside the Space
   repo specifically* — Spaces reads the YAML front matter at the top of
   `README.md` for the Space card (title, emoji, SDK). Keep the GitHub
   repo's own `README.md` as the full portfolio README; don't overwrite it.
4. Upload `models/nhanes_inference_bundle.joblib` and
   `models/ft_state_dict.pt` directly through the Space's "Files" tab if
   you didn't push them via git/LFS.
5. The Space will build automatically from `requirements.txt` and launch
   `app.py`. First build takes a few minutes (installing torch/lightgbm/
   xgboost).

Once it's live, the Space URL is what you'd link in a CV or scholarship
application — e.g. `https://huggingface.co/spaces/alm03806-dev/nhanes-health-prediction`.
