# 🛡️ Insurance Fraud Detection

> Production-shaped ML service for scoring insurance claims: LightGBM behind a
> FastAPI API, with per-prediction SHAP explanations served as raw JSON and
> out-of-distribution input warnings.

![Python](https://img.shields.io/badge/Python-3.14-blue) ![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688) ![React](https://img.shields.io/badge/React-19-61dafb) ![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange)

---

## 📌 Problem Statement

Insurance fraud costs the industry billions of dollars annually. This project
builds a binary classification pipeline to detect fraudulent claims from
structured policy and incident data. The model prioritizes **recall over
precision** — missing a fraud case is far more costly than a false alarm.

---

## 📂 Dataset

- **Source:** Insurance Claims Dataset (1000 rows, 39 features)
- **Target:** `fraud_reported` (Y/N → 1/0)
- **Class distribution:** 24.7% fraud / 75.3% non-fraud
- **Key features:** incident severity, insured hobbies, auto year, capital gains, witnesses, claim amounts

---

## 🔬 Approach

The `notebooks/` directory is the exploration record. The shipped artifact is
produced by `backend/train_pipeline.py`, not by the notebooks.

### `01.EDA.ipynb` — Exploratory analysis
- Target distribution analysis
- Detection of hidden missing values (`"?"` strings in 3 columns)
- Fraud rate by categorical features (incident severity: Major Damage → 60% fraud rate)
- Correlation analysis with `fraud_reported`

### `02_preprocessing_fe.ipynb` — Preprocessing
- Dropped irrelevant and PII columns (policy number, zip code, location)
- Normalized hidden missing values (`"?"` → `NaN`) in 3 columns
- Derived `incident_year` / `policy_bind_year` from the raw date columns
- Built sklearn `Pipeline` with `ColumnTransformer` (OrdinalEncoder + SimpleImputer)
- Stratified train/test split (80/20, `random_state=42`)

> **Correction (2026-07-27):** an earlier version of this README claimed the
> features `vehicle_age` and `injury_ratio` were engineered here. They were
> **not**. That code sat in a markdown cell and never executed — the published
> model was trained without them. The consolidated pipeline in
> `backend/train_pipeline.py` faithfully reproduces the model **as it was
> actually trained**, so those features are absent there too.

### `03_lightgbm.ipynb` — Modeling
- Baseline: LightGBM with `scale_pos_weight`
- Hyperparameter optimization: Optuna TPE (150 trials, StratifiedKFold CV)
- Metric: **PR-AUC** (preferred over ROC-AUC for imbalanced datasets)
- Overfitting analysis: Train vs Test PR-AUC comparison

### `04_xgboost.ipynb` — Comparison
- XGBoost with GridSearch, evaluated against the LightGBM baseline

### `05_SHAP.ipynb` — SHAP analysis
- Global feature importance (summary/beeswarm plot)
- Local explanation (waterfall plot per prediction)
- Key insight: `incident_severity` dominates predictions (+1.83 SHAP value for Major Damage)

---

## 📊 Results

| Model | CV PR-AUC | Test PR-AUC | Test Recall | Test Precision |
|-------|-----------|-------------|-------------|----------------|
| LightGBM (baseline) | 0.712 | 0.652 | 0.86 | 0.64 |
| XGBoost (GridSearch) | 0.716 | 0.599 | 0.61 | 0.60 |

**Best model:** LightGBM with Optuna HPO
- `n_estimators=173`, `learning_rate=0.0106`, `num_leaves=37`, `min_child_samples=93`
- Train PR-AUC: 0.789 / Test PR-AUC: 0.652 (gap: 0.137)

> **The probabilities are not calibrated.** The model was trained with
> `scale_pos_weight ≈ 3.04`, which systematically inflates the minority class's
> probabilities. `fraud_probability` is a **ranking score** for prioritising
> claims, not an absolute probability. The API states this on every response
> path (`GET /model-info → probability_calibration`).

---

## 🚀 Service

The original Streamlit prototype (`app.py`) has been **removed**. It loaded three
separate artifacts (`best_model_lgb.pkl`, `preprocessor.pkl`, `defaults.pkl`)
that no longer exist — those were consolidated into a single
`backend/models/pipeline.pkl`. It remains in the git history.

**Backend — FastAPI**

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Scores one claim. Returns the probability, a risk label, all 34 SHAP contributions as raw JSON, and out-of-distribution warnings. Every field is optional; anything omitted is filled with the training median/mode. |
| `GET /model-info` | Model card data: metrics, dataset split, training ranges, defaults, measured feature influence, fairness declaration, library versions. |
| `GET /health` | Healthcheck, including the loaded model version. |

Design decisions worth calling out:

- **One pipeline, no manual encoding.** Imputation and encoding live inside
  `pipeline.pkl`; the API layer only performs schema normalisation.
- **Fail-fast at startup.** The app verifies the preprocessing contract, feature
  alignment, transform equivalence and SHAP additivity before serving. A service
  that is up but wrong is worse than one that refuses to start.
- **SHAP as JSON, not PNG.** The chart is drawn in the browser, so it stays
  interactive and the server carries no plotting stack.
- **Guardrails.** Values outside the training range are flagged rather than
  silently extrapolated — tree models cannot reach past their training range.
- **Measured, not declared.** The UI badge marking the 16 features the model
  never splits on is derived from the booster, served through the API, and never
  hard-coded in the frontend.

**Frontend — React + Vite + TypeScript + Tailwind**, with the SHAP chart rendered
via recharts and a model card page fed entirely from `/model-info`.

Deployment (Hugging Face Spaces + Netlify) is the remaining phase.

---

## ⚙️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Nebi191/insurance-fraud-detection.git
cd insurance-fraud-detection
```

**Backend** (the dataset is already in the repo at `data/insurance_claims.csv`):

```bash
pip install -r backend/requirements.txt

# Optional — reproduce the packaged artifact from the CSV.
# Deterministic: the same input yields a bit-identical pipeline.pkl.
python backend/train_pipeline.py

cd backend
python -m uvicorn app.main:app --port 8000
# Interactive API docs: http://127.0.0.1:8000/docs
python -m pytest              # 128 tests
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

The frontend defaults to `http://127.0.0.1:8000`; override it with `VITE_API_URL`.
The backend only accepts the origins listed in `ALLOWED_ORIGINS` (comma
separated, no wildcards — it refuses to start on `*`).

**Notebooks** (exploration only, loose dependency ranges):

```bash
pip install -r notebooks/requirements.txt
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | Pipeline, preprocessing, evaluation |
| LightGBM | Gradient boosting classifier |
| XGBoost | Gradient boosting (comparison, notebooks only) |
| Optuna | Bayesian hyperparameter optimization (notebooks only) |
| SHAP | Model explainability |
| FastAPI, Pydantic v2 | API layer and request validation |
| React, Vite, TypeScript, Tailwind | Frontend |
| recharts | Interactive SHAP chart |
| joblib | Model serialization |
| pytest | Backend test suite |
| matplotlib, seaborn | Visualization (notebooks only) |

---

## 📁 Project Structure

```
insurance-fraud-detection/
│
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app, endpoints, CORS, body-size limit
│   │   ├── schemas.py        # Pydantic request/response contract
│   │   ├── model.py          # artifact loading, prediction, SHAP
│   │   └── guardrails.py     # out-of-distribution detection
│   ├── models/
│   │   ├── pipeline.pkl      # preprocessor + model, one file
│   │   └── metadata.json     # metrics, defaults, training ranges, fairness
│   ├── tests/
│   ├── train_pipeline.py     # produces both artifacts
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/       # form, SHAP chart, result card
│   │   ├── pages/            # scoring page, model card page
│   │   └── api.ts, types.ts, fields.ts
│   └── package.json
│
├── notebooks/                # 01-05, exploration record
├── data/insurance_claims.csv
├── review_log/               # code review reports
└── README.md
```

---

## ⚠️ Intended Use

This is a portfolio demo. The model uses protected attributes (sex, age, marital
status) as features and **no fairness audit has been performed** — the model card
page states this openly rather than hiding it. Model output should be used to
prioritise human review, never to decline a claim on its own.

---

## 👤 Author

**Nebi** — ML Engineer | [GitHub](https://github.com/Nebi191)
