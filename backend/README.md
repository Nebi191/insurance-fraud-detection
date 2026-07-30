---
title: Insurance Fraud Detection API
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Fraud Detection — API

FastAPI service that scores a single insurance claim with a LightGBM model
and explains the decision with per-feature SHAP contributions, served as raw
JSON (never a rendered chart — the frontend draws it interactively). Values
outside the model's training range are flagged rather than silently
extrapolated.

**⚠️ The probabilities are not calibrated.** The model was trained with a
class weight (`scale_pos_weight ≈ 3.04`) that systematically inflates the
minority (fraud) class's predicted probability. `fraud_probability` is a
**ranking score** for prioritising claims for human review, not a
statistically calibrated "this claim is X% likely to be fraudulent". See
`GET /model-info -> probability_calibration` for the same statement served
back over the API, and `GET /model-info -> fairness` for a plain declaration
that the model uses protected attributes (sex, age, marital status) as
features with no fairness audit performed.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Scores one claim. Returns `fraud_probability`, `risk_level`, all SHAP contributions as raw JSON, and `out_of_distribution_warnings`. Every field is optional; anything omitted is filled with the training median/mode. Rate-limited (see below). |
| `/model-info` | GET | Model card data: metrics, dataset split, training ranges, defaults, measured feature influence, fairness declaration, library versions, risk thresholds. |
| `/health` | GET | Healthcheck, including the loaded model version. Never rate-limited — HF's own healthcheck/uptime polling depends on it always answering. |
| `/docs` | GET | Interactive OpenAPI docs (Swagger UI). |

## Configuration (Space secrets / variables)

Set these under the Space's **Settings -> Variables and secrets**. None are
required to boot — every one has a safe default — but `ALLOWED_ORIGINS`
almost certainly needs to be set for the deployed frontend to work at all.

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated list of origins allowed by CORS, e.g. `https://your-site.netlify.app`. **No wildcards** — the app refuses to start if this contains `*`. Set it to the deployed frontend's real origin (keep `http://localhost:5173` in the list too if you want local dev against the deployed API). **On Spaces this setting has no observable effect** — see the CORS note below; set it anyway, because it is what makes the allowlist real again the moment this backend moves to a host that does not rewrite CORS. |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max `/predict` requests per client per 60s window (in-process, resets on restart). `/health` and `/model-info` are not subject to this limit. |
| `SHAP_LOCK_TIMEOUT_SECONDS` | `10` | Max time a `/predict` request waits for the shared SHAP explainer lock before failing with `503` instead of hanging. |

### CORS on Spaces: the allowlist is overridden by the platform

Measured against this running Space: an origin that is **not** in
`ALLOWED_ORIGINS` still gets `Access-Control-Allow-Origin` echoing itself, on
the preflight and on the real request. The edge proxy answers `OPTIONS` before
the request reaches this application — that response carries no `server: uvicorn`
and echoes the *requested* method rather than the configured `GET, POST` — and a
permissive header is added on the way out. The identical request against the same
image running locally is rejected with `400` and no CORS header, so this is the
platform, not the app.

CORS is a **browser** mechanism that protects **users**, not servers; `curl` was
never subject to it. This API has no authentication and returns no user data, so
there is no session to ride and nothing to exfiltrate. The real consequence is
narrower: any page can embed this backend and spend its CPU. `RATE_LIMIT_PER_MINUTE`
is what bounds that — not the allowlist.

Do not describe this deployment as origin-restricted.

## Deploying (Docker SDK)

1. On Hugging Face, create a new **Space** and choose **Docker** as the SDK
   (not "Gradio" / "Streamlit" — those assume a different entry point).
2. The Space is itself a git repository. Put the **contents of this `backend/`
   directory at the Space repo's root** — `app/`, `models/`, `Dockerfile`,
   `requirements.txt`, `.dockerignore` and this `README.md` sit directly at the
   root, not nested under a `backend/` folder. `Dockerfile`'s `COPY` paths
   assume exactly this layout.

   Clone the Space and copy the files into the clone; do **not** use
   `git subtree push`. The Hub rejects binaries that LFS/Xet does not track, at
   any size, and `models/pipeline.pkl` is a plain blob in the source repository,
   so a subtree push is refused. Forcing one would also delete the Space's
   `.gitattributes` — which is the thing that makes this work, since it already
   tracks `*.pkl` through LFS and converts the file to a pointer on commit.
3. Under **Settings -> Variables and secrets**, add `ALLOWED_ORIGINS` with the
   deployed frontend's real origin (see the table above). Add
   `RATE_LIMIT_PER_MINUTE` / `SHAP_LOCK_TIMEOUT_SECONDS` only if the defaults
   need tuning.
4. HF builds the image from the `Dockerfile` and starts the container;
   `app_port: 7860` above tells the platform's proxy which port to forward.
5. Once the Space shows "Running", verify with `GET /health` and a real
   `POST /predict` call (see `README.md` at the repo root for a full request
   body example) before pointing the frontend at it.

## Design notes carried over from the rest of the project

- **One pipeline, no manual encoding.** Imputation and encoding live inside
  `models/pipeline.pkl`; the API layer only performs schema normalisation.
- **Fail-fast at startup.** The app verifies the preprocessing contract,
  feature alignment, transform equivalence and SHAP additivity before serving.
- **SHAP as JSON, not PNG.** No plotting stack ships in this image.
- **Guardrails.** Out-of-training-range values are flagged, never silently
  extrapolated.
