# 🛡️ Insurance Fraud Detection

> End-to-end ML pipeline for detecting fraudulent insurance claims using LightGBM, SHAP explainability, and a Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange)

---

## 📌 Problem Statement

Insurance fraud costs the industry billions of dollars annually. This project builds a binary classification pipeline to detect fraudulent claims from structured policy and incident data. The model prioritizes **recall over precision** — missing a fraud case is far more costly than a false alarm.

---

## 📂 Dataset

- **Source:** Insurance Claims Dataset (1000 rows, 39 features)
- **Target:** `fraud_reported` (Y/N → 1/0)
- **Class distribution:** 24.7% fraud / 75.3% non-fraud
- **Key features:** incident severity, insured hobbies, auto year, capital gains, witnesses, claim amounts

---

## 🔬 Approach

### Notebook 1 — EDA
- Target distribution analysis
- Detection of hidden missing values (`"?"` strings in 3 columns)
- Fraud rate by categorical features (incident severity: Major Damage → 60% fraud rate)
- Correlation analysis with `fraud_reported`

### Notebook 2 — Preprocessing & Feature Engineering
- Dropped irrelevant columns (policy number, zip code, location)
- Engineered features: `vehicle_age`, `policy_bind_year`, `injury_ratio`
- Built sklearn `Pipeline` with `ColumnTransformer` (OrdinalEncoder + SimpleImputer)
- Stratified train/test split (80/20, `random_state=42`)

### Notebook 3 — Modeling
- Baseline: LightGBM and XGBoost with `scale_pos_weight`
- Hyperparameter optimization: Optuna TPE (150 trials, StratifiedKFold CV)
- Metric: **PR-AUC** (preferred over ROC-AUC for imbalanced datasets)
- Overfitting analysis: Train vs Test PR-AUC comparison

### Notebook 4 — SHAP Analysis
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

---

## 🚀 Streamlit App

The app provides two modes:

**1. Manual Prediction**
- Input key claim features (incident severity, auto year, witnesses, capital gains)
- Real-time fraud probability with risk indicator (Low / Medium / High)
- SHAP waterfall plot explaining each individual prediction

**2. Coming Soon**
- Batch CSV upload for bulk predictions
- Model metrics dashboard with PR curve and confusion matrix

---

## ⚙️ How to Run

```bash
# 1. Clone the repository
git https://github.com/Nebi191/insurance-fraud-detection.git
cd insurance-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Place insurance_claims.csv in the root directory

# 4. Run notebooks in order
# NB1_EDA.ipynb
# NB2_Preprocessing.ipynb
# NB3_Modeling.ipynb
# NB4_SHAP.ipynb

# 5. Launch the app
streamlit run app.py
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | Pipeline, preprocessing, evaluation |
| LightGBM | Gradient boosting classifier |
| XGBoost | Gradient boosting (comparison) |
| Optuna | Bayesian hyperparameter optimization |
| SHAP | Model explainability |
| Streamlit | Web application |
| joblib | Model serialization |
| matplotlib, seaborn | Visualization |

---

## 📁 Project Structure

```
insurance-fraud-detection/
│
├── NB1_EDA.ipynb
├── NB2_Preprocessing.ipynb
├── NB3_Modeling.ipynb
├── NB4_SHAP.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 👤 Author

**Nebi** — Aspiring ML Engineer | [GitHub](https://github.com/Nebi191)