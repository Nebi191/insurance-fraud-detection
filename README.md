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

---

## 🌐 Deployment

Deploy the **backend first** — the frontend needs its URL at build time.

**Backend — Hugging Face Spaces (Docker)**

Spaces builds from the root of the Space repository, so the contents of
`backend/` go to the Space root (not the `backend/` folder itself). Both files
Spaces needs are already there: `backend/Dockerfile` and a `backend/README.md`
whose YAML frontmatter declares `sdk: docker` and `app_port: 7860`.

1. Create a new Space, **SDK: Docker**, blank template.
2. Clone the Space and copy the contents of `backend/` into the clone, then
   commit and push. `README.md` must land at the Space root — Spaces reads its
   frontmatter as the build config.

   Do **not** use `git subtree push` for this. The Hub rejects binary files that
   are not tracked by LFS/Xet regardless of size, and `models/pipeline.pkl` is a
   plain blob in this repository — a subtree push is rejected outright. Worse, a
   forced one would delete the Space's own `.gitattributes`, which is exactly
   what makes the push work: it already tracks `*.pkl` through LFS, so copying
   the file into a clone of the Space converts it to a pointer automatically.
3. **Settings -> Variables and secrets:** set `ALLOWED_ORIGINS` to the Netlify
   origin, e.g. `https://<site>.netlify.app` (scheme included, no trailing
   slash, no wildcard — the app refuses to start on `*`). Optional:
   `RATE_LIMIT_PER_MINUTE` (default 30) and `SHAP_LOCK_TIMEOUT_SECONDS`
   (default 10). See the CORS caveat below for what this setting does and does
   not achieve on this particular platform.

The image installs `requirements.txt` only — never `requirements-dev.txt`; no
test or lint package belongs in a production image. It runs as a non-root user
and strips the write bit from `models/`, because `joblib.load()` reads an
unsigned pickle and unpickling is equivalent to executing code: an artifact that
the serving process could overwrite would turn any write primitive into code
execution on the next restart.

`libgomp1` is installed explicitly. LightGBM's compiled backend links against
OpenMP, which `python:*-slim` strips, and without it `import lightgbm` fails
outright — a container that dies at startup, not a subtle degradation.

**Free-tier caveat:** a Space sleeps after 48 hours of inactivity and takes
~30–60 s to wake. The frontend detects a slow first request and says so instead
of showing an unexplained spinner.

**CORS caveat — the allowlist does not take effect on Spaces.** Measured against
the live deployment: an origin that is *not* in `ALLOWED_ORIGINS` still receives
`Access-Control-Allow-Origin` echoing itself, on both the preflight and the real
request. The Spaces edge proxy answers `OPTIONS` before it reaches the
application (the preflight response carries no `server: uvicorn`, and it echoes
the requested method instead of the configured `GET, POST`) and adds a
permissive header to the response on the way out. Locally the same request is
rejected with `400` and no CORS header at all, so this is platform behaviour,
not a bug in the app.

What this does and does not mean. CORS is enforced by the *browser* and protects
*users*, not servers — `curl` never consulted it in the first place. This API has
no authentication and returns no user data, so there is no session to ride and
nothing to exfiltrate. The practical consequence is narrower: any page on the
internet can embed this backend and spend its CPU. What actually limits that is
the rate limiter, not the allowlist.

Keep `ALLOWED_ORIGINS` set anyway. The code is not wrong — the allowlist works as
written on any host that does not rewrite CORS (Cloud Run, a VPS, local Docker),
and leaving it correct means the guarantee returns the moment the backend moves.
Treat it as defence in depth that this particular platform neutralises, and do
not describe the deployed demo as origin-restricted.

**Frontend — Netlify**

**Frontend — Netlify**

This is a monorepo — Netlify is pointed at the whole repository, and the root
`netlify.toml` tells it to treat `frontend/` as the site:

```toml
[build]
  base = "frontend"        # descend into frontend/ before building
  command = "npm run build"
  publish = "dist"         # -> frontend/dist, resolved relative to `base`
```

To connect the site:

1. Netlify -> **Add new site -> Import an existing project**, pick this repo.
   Netlify reads `netlify.toml` from the repo root automatically; the `base`
   key above means the build/publish settings below apply even though the
   file sits outside `frontend/`.
2. **Site configuration -> Environment variables -> Add a variable:**
   `VITE_API_URL` = the deployed backend's URL, e.g.
   `https://<hf-username>-<space-name>.hf.space` (no trailing slash needed).
   See `frontend/.env.example` for the same value, documented for local use.
   **This is a build-time variable** — Vite inlines `import.meta.env.VITE_API_URL`
   into the bundled JS while `vite build` runs; changing it in Netlify's
   dashboard has no effect until the next deploy re-runs the build. Locally,
   leaving it unset falls back to `http://127.0.0.1:8000`.
3. Deploy. Netlify picks up `netlify.toml`'s `NODE_VERSION` automatically.

**Why the SPA needs an explicit rewrite:** the app has two routes
(`/` and `/model-card`, see `frontend/src/router.tsx`) but is a single static
`index.html` — there is no `model-card.html` on disk. Netlify serves static
files by default, so opening `/model-card` directly (or refreshing on it)
would 404 without a fallback rule. `netlify.toml` declares:

```toml
[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200   # a REWRITE, not a redirect — the URL in the address bar must not change
```

(`status = 200` matters: a 301/302 would bounce the browser to a different URL
instead of serving `index.html`'s content while `/model-card` stays in the
address bar, which is what the client-side router expects on load.)

This lives in `netlify.toml` rather than a `frontend/public/_redirects` file —
both work, but putting it beside the `base`/`command`/`publish` settings that
already have to live in this file keeps every Netlify-specific decision in one
place instead of two files that could quietly drift apart.

`netlify.toml` also sets a few static security headers (`X-Frame-Options`,
`X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`) that don't
restrict anything the app relies on. It deliberately does **not** set a
Content-Security-Policy: this app sets inline `style` attributes (React's
`style` prop, used in three components, plus recharts' SVG output), so a real
`style-src` would need `'unsafe-inline'` to avoid breaking the SHAP chart and
result card — which gives up most of the protection a CSP is meant to add. An
unverified, broken CSP is worse than none, so it was left out rather than
shipped half-checked.

**Cold start, honestly disclosed:** the backend's free-tier host sleeps after
inactivity; the first request after a sleep takes roughly 30-60 seconds
instead of instant. Rather than let that look like a hang, the UI switches to
an explicit "Waking the scoring service…" message once a request has been
running longer than a few seconds (see `frontend/src/coldStart.ts` for the
threshold and the reasoning) — both on the initial `/model-info` load and on
`/predict`, since a tab left open across the inactivity window can hit a
re-slept container on submit too.

---

## ⚙️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Nebi191/insurance-fraud-detection.git
cd insurance-fraud-detection
```

**Backend** (the dataset is already in the repo at `data/insurance_claims.csv`):

```bash
# requirements-dev.txt pulls requirements.txt in too (`-r requirements.txt`),
# so this one command installs both runtime and test/lint dependencies.
# The Docker image (Phase 7) installs ONLY requirements.txt — no test package
# belongs in a production image.
pip install -r backend/requirements-dev.txt

# Optional — reproduce the packaged artifact from the CSV.
# Deterministic: the same input yields a bit-identical pipeline.pkl.
python backend/train_pipeline.py

cd backend
python -m uvicorn app.main:app --port 8000
# Interactive API docs: http://127.0.0.1:8000/docs
python -m pytest              # 147 tests
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
