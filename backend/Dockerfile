# syntax=docker/dockerfile:1
#
# Hugging Face Spaces (Docker SDK) image for the FastAPI backend.
#
# LAYOUT ASSUMPTION: this Dockerfile is written to run with the CONTENTS of
# `backend/` copied to the Space repo ROOT (HF Spaces builds from the repo
# root, and Netlify already owns `frontend/` separately in this monorepo — see
# `backend/README.md` for the exact "what goes where" deploy steps). Paths
# below are therefore `app/`, `models/`, `requirements.txt`, NOT
# `backend/app/` etc.

# --- Base image -------------------------------------------------------------
# PINNED to 3.14, not just "3.14-slim generically": `models/pipeline.pkl` is a
# joblib/pickle file produced with CPython 3.14.6 (see requirements.txt's
# comment). A different minor version may still open the pickle, but that is
# not guaranteed, and a silent behavioural drift in a security-relevant
# artifact (a pickle IS code) is the worst possible way to find out.
FROM python:3.14-slim

WORKDIR /app

# --- System dependency: libgomp1 --------------------------------------------
# LightGBM's compiled backend is linked against libgomp (OpenMP) for its
# multi-threaded tree code. `python:*-slim` images strip it to keep the image
# small, and `import lightgbm` FAILS at import time with
# `OSError: libgomp.so.1: cannot open shared object file` without it — not a
# lazily-triggered failure, an immediate one, so this step cannot be skipped
# or deferred.
# `--no-install-recommends` + removing the apt lists afterwards keeps this to
# the one shared library instead of a full apt dependency tree.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (cached layer) -------------------------------------
# Copied and installed BEFORE the application code: this layer only
# invalidates when requirements.txt itself changes, so an ordinary code change
# does not force a multi-minute dependency reinstall on every rebuild.
#
# ONLY requirements.txt — NOT requirements-dev.txt. No test package (pytest,
# ruff, ...) belongs in a production image: it only adds image size and
# attack surface for tools that are never invoked here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code + model artifact --------------------------------------
COPY app/ ./app/
COPY models/ ./models/

# --- Non-root user (HF Spaces requirement) ----------------------------------
# HF Spaces containers run as uid 1000, not root. A dedicated user is created
# explicitly rather than relying on that convention silently: if the platform
# ever changes, `USER appuser` still enforces "not root" on its own.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser

# --- pipeline.pkl must not be writable at runtime (Phase 6 security item) ---
# `joblib.load()` reads an UNSIGNED pickle, and unpickling arbitrary data is
# equivalent to executing arbitrary code. If the artifact were writable by the
# process that serves requests, any code-execution bug in that process (or a
# bug anywhere upstream that lets an attacker write into the container's
# filesystem) could plant a malicious pickle that a RESTART would then load
# and execute with full application privileges. Root still owns the files
# (chmod happens before `USER appuser` below) and they carry no write bit for
# anyone else, so `appuser` — the identity every request actually runs as —
# cannot modify them.
RUN chmod -R a-w /app/models

# The application does not write anywhere at runtime (no cache directory, no
# log file, no upload path), so no writable directory is created for
# `appuser` — there is deliberately nothing to grant.
USER appuser

# --- Port --------------------------------------------------------------------
# HF Spaces' Docker SDK routes external traffic to the port declared in the
# Space's `README.md` frontmatter (`app_port`), NOT to `$PORT` — Spaces does
# not inject a `$PORT` environment variable the way some other Docker
# platforms (e.g. Heroku, Render) do. 7860 is used here and MUST match
# `app_port: 7860` in `backend/README.md`'s frontmatter and the `--port` value
# below; a mismatch means the platform's proxy never reaches the process.
EXPOSE 7860

# --no-reload: reload watches the filesystem for changes and is a development
# feature; there is no source to watch for changes in a built image, and the
# extra watcher thread is a pure cost in production.
#
# --workers 1 (i.e. no --workers flag, uvicorn's default is already 1): the
# HF Spaces free tier gives 2 vCPUs, and every `/predict` call already
# serialises its SHAP half through `ModelBundle._shap_lock` (see
# `app/model.py`) — a shared in-process lock only provides its correctness
# guarantee (bitwise-identical concurrent predictions) WITHIN one process.
# Running multiple worker PROCESSES would give each its own lock and its own
# copy of the model in memory, multiplying memory use on an already
# constrained tier without adding real concurrency for the one endpoint that
# is actually CPU-bound. FastAPI's synchronous `def` endpoints already run in
# a per-process thread pool (see `app/main.py`), which is the concurrency
# mechanism this deployment relies on instead.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
