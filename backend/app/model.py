"""Artifact loading, schema normalisation, prediction and SHAP.

WHAT IT DOES, AND WHY IT DOES IT THIS WAY
-----------------------------------------

1) THE ARTIFACT PATH IS FIXED AND NEVER COMES FROM USER INPUT
   `MODELS_DIR` is derived from `__file__` alone. Taking the path from an
   environment variable or from a request means path traversal / arbitrary
   pickle loading — and loading a pickle is equivalent to executing code. A
   single hard-coded path closes that entire class of attack.

2) LOADED ONCE, FAIL-FAST
   `ModelBundle.load()` is called once at application startup (FastAPI
   `lifespan`). If a file is missing, or the artifact disagrees with the
   metadata, the application does NOT start at all. A service that is "up but
   returns 500 to every request" is worse than one that never starts: the
   healthcheck looks green and the problem surfaces long after the deploy.

   `shap.TreeExplainer` is also constructed once here. Constructing it per
   request would mean re-parsing the tree structure every time (expensive and
   pointless).

3) THE FOUR STEPS THAT LIVE OUTSIDE THE PIPELINE (preprocessing_contract)
   `pipeline.pkl` covers imputation and encoding, but it does NOT cover the step
   from the raw schema to the pipeline input. `metadata.preprocessing_contract`
   lists those steps machine-readably. Their status in the API layer:

     drop_columns          -> TRIVIAL. The columns to drop (PII, raw dates,
                              _c39, auto_model) are simply ABSENT from the API
                              schema. A client cannot send them; `extra="forbid"`
                              rejects the attempt.
     derive_year_features  -> TRIVIAL. The API accepts `incident_year` /
                              `policy_bind_year` directly rather than raw dates,
                              so there is nothing to derive (see K2).
     question_mark_to_nan  -> GENUINELY APPLIED, inside `prepare_row()`.
     order_columns         -> GENUINELY APPLIED, inside `prepare_row()`.

   If either of the last two is skipped, the failure is SILENT: the "?" string
   gets treated by the encoder as an unknown category and encoded to -1 (the
   imputer never runs), and a wrong column order produces a prediction that looks
   at entirely the wrong features. Neither raises an exception — which is why the
   contract lives in the artifact rather than in code, and is verified at startup
   by `_verify_contract()`.

4) MANUAL ENCODING IS STILL FORBIDDEN
   Nothing in this file turns a categorical value into a number. All it does is
   SCHEMA NORMALISATION (fill missing fields, "?" -> NaN, order the columns).
   Turning values into numbers is, from start to finish, the `Pipeline`'s job.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import threading
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.pipeline import Pipeline

from app.guardrails import Guardrail

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# backend/app/model.py -> backend/models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# In the source dataset, "unknown" is encoded with this string.
QUESTION_MARK = "?"

# The model's positive class (fraud_reported == 1 -> "Y").
POSITIVE_CLASS_LABEL = 1

# --- Risk thresholds (K6) -------------------------------------------------- #
#
# CAUTION — THE PROBABILITIES ARE NOT CALIBRATED.
# The model was trained with `scale_pos_weight ~= 3.04`: the minority (fraud)
# class was weighted 3x. That improves PR-AUC but INFLATES the `predict_proba`
# output. So 0.71 does NOT mean "this claim is 71% likely to be fraud"; it only
# means "ranked high relative to other claims".
#
# The thresholds below are therefore a TRIAGE policy, not a statistical
# probability cut. They are kept as module-level constants so they are easy to
# change: if the business rule moves, it moves in one place (and /model-info
# reads and publishes these values from here, so the frontend keeps no copy).
RISK_THRESHOLD_MEDIUM = 0.35
RISK_THRESHOLD_HIGH = 0.65
RISK_RULE_DESCRIPTION = (
    f"low: p < {RISK_THRESHOLD_MEDIUM} | "
    f"medium: {RISK_THRESHOLD_MEDIUM} <= p < {RISK_THRESHOLD_HIGH} | "
    f"high: p >= {RISK_THRESHOLD_HIGH}"
)
CALIBRATION_WARNING = (
    "The probabilities are NOT calibrated. The model was trained with an "
    "imbalanced class weight (scale_pos_weight ~= 3.04), which systematically "
    "inflates the probabilities of the minority class. fraud_probability should "
    "be read as a RANKING score for prioritising claims, not as an absolute "
    "probability. If an absolute probability is required, the model must be "
    "calibrated separately (e.g. isotonic / Platt)."
)

# The steps we expect inside
# `preprocessing_contract.caller_must_apply_before_predict`. If the artifact is
# retrained and a new step appears (e.g. a new derived feature), the API fails at
# startup rather than skipping it silently.
EXPECTED_CONTRACT_STEPS = {
    "drop_columns",
    "derive_year_features",
    "question_mark_to_nan",
    "order_columns",
}


class ArtifactError(RuntimeError):
    """The artifact is missing or disagrees with the metadata — do not start."""


# --------------------------------------------------------------------------- #
# `_shap_lock` wait timeout (Phase 7, reviewer C2 follow-up)
# --------------------------------------------------------------------------- #
#
# `_shap_lock` serialises SHAP explanations because `shap`'s LightGBM branch
# mutates `explainer.expected_value` on every call (see `ModelBundle.__init__`).
# That lock has no bound on its own: if enough concurrent `/predict` requests
# queue up, every one of them blocks forever waiting its turn, and the client
# never times out either — it just hangs. The rate limiter in `main.py` caps
# how many requests from ONE client can even reach this point, but distinct
# clients (or a single client racing multiple browser tabs) are not capped by
# it, so an unbounded wait here is still reachable.
#
# The fix bounds the WAIT only, never the computation: once the lock is
# acquired, the explanation always finishes uninterrupted (this matters for
# `test_concurrent_predictions_are_bitwise_identical`, which depends on every
# request completing a full, uncut critical section).

SHAP_LOCK_TIMEOUT_ENV = "SHAP_LOCK_TIMEOUT_SECONDS"

# A single explanation takes about a millisecond (see `ModelBundle.__init__`),
# so under normal load the queue drains almost instantly. 10s is generous
# enough to absorb a burst without false positives in the test suite or in a
# lightly loaded deploy, while still failing a pathologically long queue fast
# instead of hanging the request indefinitely.
DEFAULT_SHAP_LOCK_TIMEOUT_SECONDS = 10.0


class ShapLockConfigurationError(ValueError):
    """`SHAP_LOCK_TIMEOUT_SECONDS` is set but is not a positive number."""


class ShapLockTimeoutError(RuntimeError):
    """The shared SHAP explainer lock could not be acquired before the timeout.

    Raised from inside a request, not at startup — `main.py` maps it to a 503
    with a `Retry-After` header rather than letting the request hang or 500.
    """


def get_shap_lock_timeout_seconds() -> float:
    """Reads `SHAP_LOCK_TIMEOUT_SECONDS`, falling back to the default.

    Read PER CALL (not cached at startup) so tests can shrink the timeout with
    `monkeypatch.setenv` for a single request without reloading the whole
    artifact — loading `pipeline.pkl` and constructing the `TreeExplainer` is
    expensive and scoped to session lifetime (see `tests/conftest.py`).
    """
    raw = os.getenv(SHAP_LOCK_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_SHAP_LOCK_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ShapLockConfigurationError(
            f"{SHAP_LOCK_TIMEOUT_ENV} must be a number, got {raw!r}."
        ) from exc
    if value <= 0:
        raise ShapLockConfigurationError(
            f"{SHAP_LOCK_TIMEOUT_ENV} must be positive, got {value!r}."
        )
    return value


# --------------------------------------------------------------------------- #
# Normalising the SHAP output shape (K7)
# --------------------------------------------------------------------------- #


def _positive_class_index(model: Any) -> int:
    """Index of the positive class (`1`) within `classes_`.

    We do not hard-code `1`: the class order depends on the labels the estimator
    was shown.

    NO SILENT FALLBACK (Codex C-2). The previous version fell back to "the last
    class" when the label could not be found. That assumption is usually right
    for binary problems but was dangerous HERE: a wrong index shifts both the
    `predict_proba` column and the SHAP axis at the same time, so
    `_verify_shap_additivity` would use the same wrong axis and the identity
    would STILL HOLD. In other words the bug would slip past its own verification
    and bring up a service that was "sign-flipped but self-consistent".

    Because the artifact maps `fraud_reported` to 0/1, `classes_` is always
    `[0, 1]` in this project; reaching this error means the artifact changed
    unexpectedly, and the correct behaviour is not to start.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ArtifactError(
            "The model has no `classes_`, so the positive class index cannot be "
            "determined. The artifact may not be a scikit-learn classifier."
        )
    classes = np.asarray(classes)
    matches = np.flatnonzero(classes == POSITIVE_CLASS_LABEL)
    if not matches.size:
        raise ArtifactError(
            f"The positive class ({POSITIVE_CLASS_LABEL}) is not among the "
            f"model's class labels: classes_={classes.tolist()}. Which column "
            "means 'fraud' cannot be guessed; continuing on an assumption would "
            "silently invert the probability and the SHAP signs."
        )
    return int(matches[0])


def _select_positive_class(values: Any, base_values: Any, positive_index: int) -> tuple[np.ndarray, float]:
    """Picks the positive class's contributions from the SHAP output WITHOUT
    assuming a shape.

    For binary classification `TreeExplainer` may return different shapes
    depending on the version and backend; we reduce all of them to one form:

        list/tuple            -> one array per class, [positive_index] is taken
        ndarray (n, f, c)     -> last axis is class, [..., positive_index]
        ndarray (n, f)        -> already the positive class log-odds in a binary
                                 model

    In this setup (shap 0.52 + lightgbm 4.6, objective='binary') the measured
    shape is (1, 34) with a scalar base_value — but indexing the wrong axis would
    be a bug that silently inverts the signs, so we check the shape. That the
    choice is CORRECT is proven separately by additivity (see
    `_verify_shap_additivity`).
    """
    if isinstance(values, (list, tuple)):
        values = values[positive_index]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:  # (n_rows, n_features, n_classes)
        array = array[..., positive_index]
    if array.ndim == 2:  # (n_rows, n_features) -> we want a single row
        array = array[0]
    if array.ndim != 1:
        raise ArtifactError(f"Unexpected SHAP value shape: {np.asarray(values).shape}")

    if isinstance(base_values, (list, tuple)):
        base_values = base_values[positive_index]
    base = np.asarray(base_values, dtype=float)
    if base.ndim == 0:
        base_value = float(base)
    elif base.ndim == 1:
        # We predict a single row: size 1 means the row axis, otherwise the class axis.
        base_value = float(base[0]) if base.size == 1 else float(base[positive_index])
    else:  # (n_rows, n_classes)
        base_value = float(base[0, positive_index])

    return array, base_value


# --------------------------------------------------------------------------- #
# Risk label
# --------------------------------------------------------------------------- #


def classify_risk(probability: float) -> str:
    """Maps a probability to a `low` / `medium` / `high` triage label."""
    if probability >= RISK_THRESHOLD_HIGH:
        return "high"
    if probability >= RISK_THRESHOLD_MEDIUM:
        return "medium"
    return "low"


# --------------------------------------------------------------------------- #
# ModelBundle
# --------------------------------------------------------------------------- #


class ModelBundle:
    """Holds the loaded pipeline, metadata and SHAP explainer together.

    Why one object: the three are coupled (the explainer points at the pipeline's
    model, the feature names point at the metadata). Keeping them in separate
    globals would be a design that invites updating one and forgetting another.
    """

    def __init__(self, pipeline: Pipeline, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata

        self.input_order: list[str] = metadata["feature_list"]["pipeline_input_order"]
        self.defaults: dict[str, Any] = {
            name: spec["value"] for name, spec in metadata["defaults"].items()
        }
        self.display_names: list[str] = metadata["feature_list"]["transformed_display_names"]

        self._preprocessor = pipeline.named_steps["preprocessor"]
        self._model = pipeline.named_steps["model"]
        self._positive_index = _positive_class_index(self._model)

        # The `question_mark_to_nan` columns are NOT HARD-CODED; they are read
        # from the contract. If the model is retrained and that list changes, the
        # API warns automatically.
        self.question_mark_columns: list[str] = list(
            _contract_step(metadata, "question_mark_to_nan")["columns"]
        )

        # Phase 2 — OOD checking. The ranges are not embedded in the guardrail;
        # they are read from the metadata, so retraining the model updates the
        # thresholds by itself (see `guardrails.py`).
        self.guardrail = Guardrail(metadata)

        self.explainer = shap.TreeExplainer(self._model)

        # shap's LightGBM branch REASSIGNS `explainer.expected_value` on every
        # call. Even though the value is always the same, that is a mutation of
        # shared state; because FastAPI runs synchronous endpoints in a thread
        # pool, concurrent requests hit the same explainer. Explaining a single
        # row takes about a millisecond, so the cost of the lock is negligible —
        # preferable to an undefined race.
        self._shap_lock = threading.Lock()

    # -- loading ------------------------------------------------------------ #

    @classmethod
    def load(cls) -> ModelBundle:
        """Loads the artifacts from the fixed path and verifies their consistency."""
        missing = [p.name for p in (PIPELINE_PATH, METADATA_PATH) if not p.is_file()]
        if missing:
            raise ArtifactError(
                f"Model artifact not found: {missing} (searched in: {MODELS_DIR}). "
                "Run `python backend/train_pipeline.py` first."
            )

        pipeline = joblib.load(PIPELINE_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        bundle = cls(pipeline, metadata)
        bundle._verify_contract()
        bundle._verify_feature_alignment()
        bundle._verify_transform_equivalence()
        bundle._verify_shap_additivity()
        bundle._warn_on_library_version_drift()
        return bundle

    def _verify_transform_equivalence(self) -> None:
        """`preprocessor.transform` + `model.predict_proba` == `pipeline.predict_proba`.

        For performance, `predict()` splits the pipeline into two steps (see
        `_transform`). At startup we prove that this shortcut produces a
        BIT-IDENTICAL result to the full Pipeline call. Exact equality, not
        `pytest.approx`: even the smallest deviation means the split is wrong
        (e.g. this check fires if a third step is ever added to the pipeline).
        """
        frame = self.prepare_row({})
        via_pipeline = self.pipeline.predict_proba(frame)
        via_steps = self._model.predict_proba(self._transform(frame))
        if not np.array_equal(via_pipeline, via_steps):
            raise ArtifactError(
                "The preprocessor+model split does not match pipeline.predict_proba: "
                f"{via_pipeline!r} != {via_steps!r}. The pipeline may contain an "
                "unexpected step; the predict() shortcut is not safe."
            )

    def _warn_on_library_version_drift(self) -> None:
        """WARNS when the runtime versions drift from the training versions.

        WHY WE DO NOT FAIL: `pipeline.pkl` is a pickle, and scikit-learn /
        lightgbm / numpy classes are serialised inside it. A different version
        usually loads fine, sometimes behaves differently in silence, and rarely
        fails to load at all. Refusing to start on every drift would needlessly
        block the Phase 7 deploy (Docker image). But staying silent is not an
        option either: silent drift is precisely the most expensive form of
        "why is the model scoring differently?".

        The warning is NOT added to the `/health` response — the API contract
        stays narrow; this is an operator signal, not client data.
        """
        recorded: dict[str, str] = self.metadata.get("library_versions", {})
        runtime = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        }
        drift = {
            name: (recorded[name], runtime[name])
            for name in runtime
            if name in recorded and recorded[name] != runtime[name]
        }
        if drift:
            details = ", ".join(
                f"{name}: trained={trained} runtime={current}"
                for name, (trained, current) in sorted(drift.items())
            )
            logger.warning(
                "Library version drift detected (%d packages). pipeline.pkl was not "
                "produced with these versions; predictions may differ silently. %s",
                len(drift),
                details,
            )

    def _verify_contract(self) -> None:
        """Verifies that the contract steps are exactly the ones we expect."""
        steps = {
            step["step"]
            for step in self.metadata["preprocessing_contract"]["caller_must_apply_before_predict"]
        }
        if steps != EXPECTED_CONTRACT_STEPS:
            raise ArtifactError(
                "The preprocessing_contract steps differ from what was expected. "
                f"expected={sorted(EXPECTED_CONTRACT_STEPS)} found={sorted(steps)}. "
                "The API layer cannot silently skip a preparation step it does not know."
            )

        # The `order_columns` contract must agree with the actual input order.
        ordered = _contract_step(self.metadata, "order_columns")["columns"]
        if list(ordered) != list(self.input_order):
            raise ArtifactError("order_columns does not agree with pipeline_input_order.")

        unknown_qm = set(self.question_mark_columns) - set(self.input_order)
        if unknown_qm:
            raise ArtifactError(f"Unknown column(s) in question_mark_to_nan: {sorted(unknown_qm)}")

    def _verify_feature_alignment(self) -> None:
        """Do defaults / transformed names / column order all agree?"""
        if set(self.defaults) != set(self.input_order):
            raise ArtifactError("metadata.defaults is not the same column set as pipeline_input_order.")

        transformed = list(self._preprocessor.get_feature_names_out())
        if transformed != list(self.metadata["feature_list"]["transformed_order"]):
            raise ArtifactError(
                "The transformed column order produced by the pipeline does not match the metadata."
            )
        if len(transformed) != len(self.display_names):
            raise ArtifactError(
                "transformed_display_names length does not match transformed_order: "
                f"{len(self.display_names)} != {len(transformed)}"
            )
        # The cat__/remainder__ prefixes MUST NOT leak into the SHAP names (K7).
        leaked = [n for n in self.display_names if n.startswith(("cat__", "remainder__"))]
        if leaked:
            raise ArtifactError(f"Raw prefix left in transformed_display_names: {leaked}")

        # NAME ALIGNMENT IS VERIFIED BY POSITION (Codex C-3).
        #
        # The two checks above only look at LENGTH and PREFIX; neither says
        # anything about the ORDER of the names. If the metadata list were
        # permuted (same names, different order), the application would start
        # cleanly and the scores would stay correct — but `/predict` would
        # attribute every SHAP contribution to the WRONG feature, which means the
        # very thing this demo is selling would be lying silently.
        #
        # `_verify_shap_additivity` cannot catch this: the sum is unchanged, so
        # the identity still holds. The tests did catch it (see
        # `test_shap_feature_names_follow_the_transformed_column_order`) but
        # startup did not; a broken artifact could therefore reach production.
        # The fix is to derive the names from the pipeline's own output rather
        # than TRUSTING the metadata, and compare.
        expected_display = [name.split("__", 1)[-1] for name in transformed]
        if self.display_names != expected_display:
            mismatches = [
                f"position {position}: metadata={actual!r} pipeline={expected!r}"
                for position, (actual, expected) in enumerate(
                    zip(self.display_names, expected_display, strict=True)
                )
                if actual != expected
            ]
            raise ArtifactError(
                "transformed_display_names does not match the pipeline's column order "
                f"({len(mismatches)} positions). SHAP contributions would have been "
                f"published under the wrong feature names. First differences: {mismatches[:5]}"
            )

    def _verify_shap_additivity(self) -> None:
        """Proves mathematically that the positive-class selection is CORRECT.

        SHAP's fundamental identity: sum(shap) + base_value = the model's raw
        margin (log-odds). Had we picked the wrong class or axis, that identity
        would break. We check it once at startup on the default row — seeing the
        bug here as a startup error is far cheaper than seeing it in production
        as 34 sign-flipped SHAP values.
        """
        frame = self.prepare_row({})
        transformed = self._transform(frame)
        probability = float(self._model.predict_proba(transformed)[0, self._positive_index])
        values, base_value = self._shap_for_transformed(transformed)

        margin = float(values.sum() + base_value)
        # For LightGBM's binary objective, sigmoid(margin) equals predict_proba.
        reconstructed = 1.0 / (1.0 + math.exp(-margin))
        if abs(reconstructed - probability) > 1e-6:
            raise ArtifactError(
                "SHAP additivity check failed: "
                f"sigmoid(sum(shap)+base)={reconstructed:.9f} != predict_proba={probability:.9f}. "
                "The wrong class axis was most likely selected."
            )

    # -- input preparation -------------------------------------------------- #

    def prepare_row(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Converts a validated request body into the single-row DataFrame the
        pipeline expects.

        The `payload` keys are metadata column names (thanks to Pydantic's
        `model_dump(by_alias=True)`, `capital-gains` arrives hyphenated).

        Contract steps applied here:
          * question_mark_to_nan  -> a field sent as "?" becomes NaN
          * order_columns         -> columns follow `pipeline_input_order`

        TWO DIFFERENT KINDS OF "MISSING" (K3 vs K4):
          field never sent / null  -> the metadata.defaults value (median/mode)
          field sent as "?"        -> NaN, handled by the pipeline's SimpleImputer
        In this model both can end up at the same value (the imputer also uses the
        mode), but they mean different things, and if the filling strategy changes
        the behaviour diverges correctly.
        """
        record: dict[str, Any] = {}
        for column in self.input_order:
            value = payload.get(column)
            if value is None:
                # K3 — field absent: median/mode from the training set.
                value = self.defaults[column]
            elif column in self.question_mark_columns and value == QUESTION_MARK:
                # K4 — field is "unknown": hand it to the pipeline's imputer.
                value = np.nan
            record[column] = value

        # Building with `columns=` guarantees the `order_columns` step.
        return pd.DataFrame([record], columns=self.input_order)

    # -- prediction --------------------------------------------------------- #

    def _shap_for_transformed(self, transformed: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Positive-class SHAP contributions + base value for a transformed row.

        It takes the ALREADY TRANSFORMED matrix rather than the raw row: that way
        the matrix the model scores and the matrix the explainer explains are the
        same object — "they must be the same" becomes a structural property of
        the code rather than a comment.
        """
        timeout = get_shap_lock_timeout_seconds()
        if not self._shap_lock.acquire(timeout=timeout):
            raise ShapLockTimeoutError(
                f"Timed out after {timeout:g}s waiting for the shared SHAP explainer "
                "lock; too many concurrent /predict requests are queued ahead of "
                "this one."
            )
        try:
            # The `explainer(X)` (Explanation API) form is preferred:
            # `shap_values()` emits a UserWarning on every call for binary
            # LightGBM models — log noise on every request. The two are
            # numerically identical.
            explanation = self.explainer(transformed)
            values, base_value = _select_positive_class(
                explanation.values, explanation.base_values, self._positive_index
            )
        finally:
            self._shap_lock.release()
        return values, base_value

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transforms the prepared row with the pipeline's OWN preprocessor step.

        THE MANUAL-ENCODING BAN IS NOT VIOLATED: what runs here is the very
        `ColumnTransformer` inside the pipeline (imputer + OrdinalEncoder). We are
        not writing our own encoding dictionary, only calling two of the
        pipeline's steps separately.

        WHY CALL THEM SEPARATELY: `pipeline.predict_proba(frame)` would run the
        preprocessor once, and SHAP would need it run a second time — the same
        transform twice per request. Transforming once and handing the same matrix
        to both the model and the explainer is not only faster, it also makes
        "what is explained is what was scored" a guarantee at the code level.

        `_verify_transform_equivalence()` proves at startup that this split
        produces a BIT-IDENTICAL result to `pipeline.predict_proba`, so the
        shortcut cannot drift silently.
        """
        return self._preprocessor.transform(frame)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Scores a single claim and explains it with SHAP contributions.

        The returned dict maps directly onto the `PredictResponse` schema; the
        HTTP layer performs no computation.
        """
        frame = self.prepare_row(payload)
        transformed = self._transform(frame)

        probability = float(self._model.predict_proba(transformed)[0, self._positive_index])
        values, base_value = self._shap_for_transformed(transformed)

        if values.shape[0] != len(self.display_names):
            raise ArtifactError(
                f"SHAP returned {values.shape[0]} values, {len(self.display_names)} were expected."
            )

        # Every feature is returned (the frontend decides how many to show),
        # sorted by descending abs(value) — the strongest contribution first.
        shap_values = [
            {"feature": name, "value": float(value), "base_value": base_value}
            for name, value in zip(self.display_names, values, strict=True)
        ]
        shap_values.sort(key=lambda item: abs(item["value"]), reverse=True)

        return {
            "fraud_probability": probability,
            "risk_level": classify_risk(probability),
            "shap_values": shap_values,
            # The OOD check runs on the RAW payload, not on `frame`: because
            # `prepare_row()` fills missing fields with defaults, the transformed
            # row no longer records "did the user actually send this?". That is
            # exactly the question the guardrail has to answer.
            "out_of_distribution_warnings": self.guardrail.check(payload),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _contract_step(metadata: dict[str, Any], name: str) -> dict[str, Any]:
    """Fetches the named step from `preprocessing_contract`."""
    for step in metadata["preprocessing_contract"]["caller_must_apply_before_predict"]:
        if step["step"] == name:
            return step
    raise ArtifactError(f"preprocessing_contract has no '{name}' step.")
