"""Phase 0 — pipeline consolidation.

This script merges the preprocessing (02_preprocessing_fe.ipynb) and model
training (03_lightgbm.ipynb) steps that were scattered across notebooks into ONE
sklearn `Pipeline`, and produces two artifacts:

    backend/models/pipeline.pkl    -> preprocessor + model, a single file
    backend/models/metadata.json   -> defaults, training_ranges, metrics, versions

Why a single Pipeline?
    In the old setup `preprocessor.pkl` and `best_model_lgb.pkl` were separate
    files and the serving layer had to call them by hand, in order. That is a
    design wide open to "skipping the transform" and "writing a manual encoding
    dictionary". With one Pipeline, *imputation + encoding* travel as an
    inseparable part of the model: the caller can no longer write its own
    encoding dictionary, and has no reason to.

The BOUNDARY of the Pipeline (important — misreading this breaks Phase 1):
    The Pipeline does NOT cover everything. The `df` in
    `pipeline.predict_proba(df)` is NOT a 39-column RAW CSV row but the
    34-column frame that has been through `load_raw_frame()`
    (`metadata.feature_list.pipeline_input_order`).

    INSIDE the Pipeline (automatic, the caller does nothing):
      - categorical NaN imputation (`SimpleImputer(strategy='most_frequent')`)
      - categorical -> numeric encoding (`OrdinalEncoder`, unknown -> -1)
      - numeric columns pass through (LightGBM carries NaN natively)

    OUTSIDE the Pipeline (inside `load_raw_frame()`; the API layer must do this
    ITSELF). These steps are also written machine-readably under
    `metadata.json -> preprocessing_contract` by `build_preprocessing_contract()`;
    Phase 1 should look at that field, not at documentation:
      1. Drop the `DROP_COLS` columns (PII + raw dates + `_c39` + `auto_model`)
      2. Derive `incident_date` -> `incident_year` and `policy_bind_date` ->
         `policy_bind_year`
      3. Normalise `"?"` -> `NaN` in `QUESTION_MARK_COLS`.
         CAUTION: if the API layer skips this, the `"?"` string enters the
         pipeline as a *valid category*; the encoder treats it as unknown and
         encodes it to -1, and the imputer NEVER runs. A silent, result-breaking
         bug.
      4. Put the columns into `pipeline_input_order`

    So `model.py` in Phase 1 cannot simply say "load the pkl and hand it the
    request"; it has to write a thin preparation layer that applies the four
    steps above. That layer DOES NO ENCODING (still forbidden) — it only performs
    schema normalisation.

Critical: the Pipeline is fitted on RAW (cleaned but NOT transformed) X_train.
Fitting on an already-transformed matrix would leave the preprocessor outside the
pipeline and defeat the entire purpose of the consolidation.

Usage:
    python backend/train_pipeline.py
    python backend/train_pipeline.py --data path/to/insurance_claims.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import shap
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# --------------------------------------------------------------------------- #
# Constants — kept identical to notebooks 02 & 03.
# --------------------------------------------------------------------------- #

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "insurance_claims.csv"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "models"

TARGET = "fraud_reported"
TARGET_MAP = {"Y": 1, "N": 0}

# PII plus columns that are meaningless to the model or carry leakage risk.
# `policy_number`, `insured_zip` and `incident_location` are direct PII, and
# because they are dropped HERE they never enter defaults/metadata either.
# `incident_date` / `policy_bind_date` are dropped in their raw form; their
# information is preserved below as `incident_year` / `policy_bind_year`.
# `_c39` is the empty trailing column in the Kaggle CSV (absent in some copies).
# `auto_model` is high cardinality and largely synonymous with `auto_make`.
DROP_COLS = [
    "policy_number",
    "insured_zip",
    "incident_location",
    "incident_date",
    "policy_bind_date",
    "_c39",
    "auto_model",
]

# Year features derived from the raw date columns. The derivation happens OUTSIDE
# the Pipeline (in load_raw_frame); the API layer has to do exactly the same.
DATE_YEAR_FEATURES = {
    "incident_date": "incident_year",
    "policy_bind_date": "policy_bind_year",
}

# In these columns, missing values are encoded as the "?" string.
QUESTION_MARK_COLS = ["property_damage", "police_report_available", "collision_type"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# The best parameters found by Optuna (notebook 03). Phase 0 is a consolidation
# phase, not a modelling phase: HPO is not re-run; we do a deterministic refit
# with the known-best parameters.
LGBM_PARAMS = {
    "n_estimators": 173,
    "learning_rate": 0.010634,
    "num_leaves": 37,
    "min_child_samples": 93,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


# --------------------------------------------------------------------------- #
# Fairness / model card — protected attributes
# --------------------------------------------------------------------------- #
#
# This block is the record of a DELIBERATE product decision: the model was
# trained with these features and stays that way (no retraining). But in an
# insurance/fintech context, deciding on the basis of a protected attribute is
# regulated; rather than carrying that silently, the artifact declares it openly.
#
# STRUCTURED data, not free text: the Phase 5 model card page reads this through
# `GET /model-info` and renders it as a table.

# Directly protected/sensitive attributes (USED as model features).
PROTECTED_ATTRIBUTES = [
    {
        "feature": "insured_sex",
        "basis": "sex",
        "severity": "high",
        "rationale": (
            "Sex is a directly protected attribute in most jurisdictions. It is "
            "used as a model feature, so the fraud score can vary with sex."
        ),
    },
    {
        "feature": "age",
        "basis": "age",
        "severity": "high",
        "rationale": (
            "Age is a protected attribute. Some jurisdictions permit its use in "
            "insurance pricing on actuarial grounds, but the same exemption "
            "cannot be assumed for scoring the *suspicion* of fraud."
        ),
    },
    {
        "feature": "insured_relationship",
        "basis": "marital_and_familial_status",
        "severity": "medium",
        "rationale": (
            "The values (husband/wife/unmarried/own-child/...) directly disclose "
            "marital and familial status, which is a protected attribute in many "
            "jurisdictions. It is also a strong proxy for sex "
            "(husband/wife -> insured_sex)."
        ),
    },
    {
        "feature": "insured_education_level",
        "basis": "socioeconomic_proxy",
        "severity": "medium",
        "rationale": (
            "Not a protected class in the classic sense, but a well-documented "
            "proxy for socioeconomic background and, indirectly, race/national "
            "origin. It should be treated as a proxy variable in a discrimination "
            "audit."
        ),
    },
]

# Not directly protected attributes, but carrying proxy risk in an audit.
PROXY_RISK_ATTRIBUTES = [
    {
        "feature": "insured_occupation",
        "basis": "socioeconomic_proxy",
        "rationale": "Occupation is a strong proxy for income, education and demographics.",
    },
    {
        "feature": "insured_hobbies",
        "basis": "lifestyle_proxy",
        "rationale": (
            "It shows high influence in SHAP; it may proxy lifestyle or cultural "
            "background, and there is no causal fraud rationale behind it."
        ),
    },
    {
        "feature": "policy_state",
        "basis": "geographic_proxy",
        "rationale": "Geographic features carry a redlining-like indirect discrimination risk.",
    },
    {
        "feature": "incident_state",
        "basis": "geographic_proxy",
        "rationale": "Geographic features carry a redlining-like indirect discrimination risk.",
    },
    {
        "feature": "incident_city",
        "basis": "geographic_proxy",
        "rationale": "Geographic features carry a redlining-like indirect discrimination risk.",
    },
]


def build_feature_influence(fitted_pipeline: Pipeline, display_names: list[str]) -> dict:
    """Each feature's MEASURED influence in the trained model.

    WHY IN THE METADATA AND NOT AT RUNTIME:
    Split/gain counts are a fact about the moment of training — the trained trees
    are fixed, and these numbers are not something to recompute on every request.
    Their proper home is next to the artifact. Computing them at runtime would be
    both wasteful and ambiguous ("which model's numbers?").

    WHY IT MATTERS:
    GIVING a feature to the model is not the same as the model USING it. If
    LightGBM never split on a column during training (`split_count` = 0), that
    column cannot affect any prediction and its SHAP contribution is always
    exactly 0.0. That distinction is critical both for the product (the user
    changes a form field and sees the result not move) and for the fairness
    declaration.

    THE KEYS ARE API FIELD NAMES:
    `display_names` already has the `cat__`/`remainder__` prefix stripped, so
    `capital-gains` arrives hyphenated — exactly the field name in the API schema.
    The frontend needs no additional mapping table.
    """
    booster = fitted_pipeline.named_steps["model"].booster_

    # The booster's own feature names and our clean names must correspond BY
    # POSITION; if they did not, every number would be written against the wrong
    # feature, and that would be a silent bug.
    booster_names = [name.split("__", 1)[-1] for name in booster.feature_name()]
    if booster_names != list(display_names):
        raise ValueError(
            "Booster feature names do not match transformed_display_names; the "
            "influence counts would be written against the wrong features."
        )

    split_counts = booster.feature_importance(importance_type="split")
    gains = booster.feature_importance(importance_type="gain")

    features: dict[str, dict] = {}
    for name, split_count, gain in zip(display_names, split_counts, gains):
        split_count = int(split_count)
        features[name] = {
            "split_count": split_count,
            "gain": float(gain),
            # Definition: did the trained trees split on this column at least once?
            "has_influence": split_count > 0,
        }

    dead = sorted(name for name, info in features.items() if not info["has_influence"])

    return {
        "description": (
            "Feature influence measured from the trained LightGBM booster. "
            "split_count = the number of splits made on this column across the "
            "trees, gain = the total gain of those splits, has_influence = "
            "split_count > 0."
        ),
        "measured_from": "lightgbm booster_.feature_importance(importance_type='split'|'gain')",
        "interpretation_note": (
            "A feature with has_influence=false cannot affect any prediction in "
            "THIS TRAINED model; its SHAP contribution is always exactly 0.0. "
            "That does NOT mean the feature was withheld from the model — it was "
            "given, and the trees did not use it. If the model is retrained "
            "(different data, seed or hyperparameters) this list may change."
        ),
        "summary": {
            "n_features": len(features),
            "n_with_influence": len(features) - len(dead),
            "n_without_influence": len(dead),
            "features_without_influence": dead,
        },
        "features": features,
    }


def build_fairness_section(
    cat_cols: list[str], num_cols: list[str], feature_influence: dict
) -> dict:
    """The machine-readable fairness section of the model card.

    Feature names are validated against the list of columns actually used in
    training, so the metadata cannot go stale by declaring a "protected attribute"
    the model does not have (or the other way round).

    DECLARATION AND MEASUREMENT ARE SEPARATED:
    This section used to carry only `used_as_model_feature: true`. Technically
    correct, but it misled the reader: the model card said "sex can affect the
    score" while the measurement said "the trained trees never split on that
    column". Now the two fields sit side by side:

        used_as_model_feature -> the feature was GIVEN to the model (declaration)
        has_influence         -> the trained trees USED it (measurement)

    The numbers come from `feature_influence`, i.e. they are computed from the
    booster; they are not hand-written constants and they update by themselves
    when the model changes.
    """
    model_features = set(cat_cols) | set(num_cols)

    for entry in (*PROTECTED_ATTRIBUTES, *PROXY_RISK_ATTRIBUTES):
        if entry["feature"] not in model_features:
            raise ValueError(
                f"The fairness declaration lists '{entry['feature']}' but it is not "
                "among the model features. The list is out of sync with training."
            )

    measured = feature_influence["features"]

    def with_measurement(entry: dict) -> dict:
        influence = measured[entry["feature"]]
        return {
            **entry,
            # DECLARATION: the feature was given to the model as an input.
            "used_as_model_feature": True,
            # MEASUREMENT: was it actually used by the trained model?
            "split_count": influence["split_count"],
            "has_influence": influence["has_influence"],
        }

    return {
        "status": "declared_not_audited",
        "field_semantics": (
            "used_as_model_feature = the feature was GIVEN to the model. "
            "has_influence = the trained trees actually split on this column. "
            "THESE ARE NOT THE SAME THING: a feature with split_count=0 cannot "
            "affect any prediction in this trained model, and its SHAP "
            "contribution is always exactly 0.0. This measurement IS NOT A "
            "SUBSTITUTE FOR A FAIRNESS AUDIT: (1) the result can change if the "
            "model is retrained, (2) proxy features (occupation, hobbies, "
            "geography) can carry the same signal back indirectly, (3) no "
            "group-level metric was computed for the attributes whose influence is "
            "not zero."
        ),
        "protected_attributes_used_as_features": [
            with_measurement(entry) for entry in PROTECTED_ATTRIBUTES
        ],
        "proxy_risk_attributes": [with_measurement(entry) for entry in PROXY_RISK_ATTRIBUTES],
        "audit_performed": False,
        "audit_metrics_computed": [],
        "intended_use": "demo_and_portfolio_only",
        "production_requirements": [
            (
                "Run a group-level fairness audit per protected attribute (e.g. "
                "demographic parity difference, equalised odds / TPR-FPR gap, "
                "calibration gap)."
            ),
            (
                "Measure whether dropping a protected attribute actually reduces "
                "discrimination: proxy features (occupation, hobbies, geography) can "
                "carry the signal back."
            ),
            (
                "Human review is mandatory: model output must be used for "
                "prioritisation/triage, never to decline a claim on its own."
            ),
            (
                "Verify compliance with local regulation (e.g. EU AI Act high-risk "
                "system obligations, US state insurance anti-discrimination law)."
            ),
        ],
        "notes": (
            "The model was trained with these features and Phase 0 deliberately "
            "left it that way: the goal was to reproduce the published model "
            "FAITHFULLY. This section is not a compliance statement but an open "
            "record of a known risk.\n\n"
            "MEASUREMENT NOTE: in this trained model some protected/proxy "
            "attributes have a split_count of 0, meaning their measured influence "
            "is zero (every SHAP contribution is exactly 0.0). That DOES NOT MEAN "
            "the model is fair, or that it does not discriminate on the attribute "
            "in question. The only thing that can be said is that the measured "
            "influence of those features in this specific set of trained trees is "
            "zero. No group-level fairness metric was computed for the attributes "
            "whose influence is not zero; audit_performed is still false and the "
            "production_requirements items below apply in full."
        ),
    }


def build_preprocessing_contract(pipeline_input_order: list[str]) -> dict:
    """The machine-readable contract for the preparation left OUTSIDE the Pipeline.

    In Phase 1 `model.py` has to apply these steps; writing them into the metadata
    closes the "the pipeline handles everything" misconception at the artifact
    level. It is derived from the code constants (never hand-written) so it can
    never drift from the code.
    """
    return {
        "summary": (
            "pipeline.pkl covers imputation and encoding; it DOES NOT cover the "
            "step from the raw CSV schema to the pipeline input. The caller (the "
            "API layer) must apply the steps below BEFORE handing anything to the "
            "pipeline."
        ),
        "inside_pipeline": [
            "categorical_imputation: SimpleImputer(strategy='most_frequent')",
            "categorical_encoding: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
            "numeric_passthrough: no imputation, LightGBM carries NaN natively",
        ],
        "caller_must_apply_before_predict": [
            {
                "step": "drop_columns",
                "columns": DROP_COLS,
                "reason": "PII, raw dates, the empty trailing column and the high-cardinality auto_model.",
            },
            {
                "step": "derive_year_features",
                "mapping": dict(DATE_YEAR_FEATURES),
                "reason": "The pipeline does not parse dates; the year feature is derived outside it.",
            },
            {
                "step": "question_mark_to_nan",
                "columns": QUESTION_MARK_COLS,
                "reason": (
                    "In these columns a missing value is encoded as the '?' string. "
                    "Without the conversion, the encoder treats '?' as an unknown "
                    "category and encodes it to -1, and the imputer never runs -> a "
                    "silently wrong prediction."
                ),
            },
            {
                "step": "order_columns",
                "columns": pipeline_input_order,
                "reason": "The pipeline input expects these column names in this order.",
            },
        ],
        "note_on_encoding_ban": (
            "These steps are SCHEMA NORMALISATION, not encoding. Turning "
            "categorical values into numbers remains the pipeline's job alone."
        ),
    }


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def load_raw_frame(csv_path: Path) -> pd.DataFrame:
    """Reads the raw CSV and applies the cleaning from notebook 02.

    CAUTION: the feature engineering designed in notebook 02 (`vehicle_age`,
    `injury_ratio` etc.) is NOT here — that code stayed in a markdown cell and
    never ran; the published model was trained without it. Phase 0 reproduces that
    model FAITHFULLY; it ADDS no new features.

    The one exception is `incident_year` / `policy_bind_year`: those are not new
    information but the model-ready form of the raw date columns, and they were
    present in the original training too. (Note: `incident_year` is constant in
    this dataset — every row is 2015 — so it adds no information to the model; the
    Phase 2 guardrail has to take that into account.)

    This entire function lives OUTSIDE the Pipeline; the API layer has to apply
    the same steps (see the module docstring + `build_preprocessing_contract`).
    """
    df = pd.read_csv(csv_path)

    # Derive the year from the date columns, then drop the raw dates.
    # These are MANDATORY: without them `incident_year`/`policy_bind_year` cannot
    # be produced and the pipeline is called with a missing column. Instead of a
    # bare `KeyError` we raise an error that names the missing column.
    missing_dates = [c for c in DATE_YEAR_FEATURES if c not in df.columns]
    if missing_dates:
        raise ValueError(
            f"Mandatory date column(s) missing from the CSV: {missing_dates}. "
            f"{[DATE_YEAR_FEATURES[c] for c in missing_dates]} are derived from them."
        )
    for date_col, year_col in DATE_YEAR_FEATURES.items():
        df[year_col] = pd.to_datetime(df[date_col]).dt.year

    present = [c for c in DROP_COLS if c in df.columns]
    absent = [c for c in DROP_COLS if c not in df.columns]
    if absent:
        print(f"[info] columns in the drop list but absent from the CSV: {absent}")
    df = df.drop(columns=present)

    # "?" -> NaN. Imputation happens inside the pipeline (SimpleImputer) so that
    # train and serve cannot behave differently.
    # The same defence as for DROP_COLS: indexing a missing column directly blows
    # up with a `KeyError`; here it is skipped and the situation is made visible.
    # (It is not swallowed silently: a missing column never enters the categorical
    # list, so the pipeline stays consistent — but the operator should know.)
    qm_present = [c for c in QUESTION_MARK_COLS if c in df.columns]
    qm_absent = [c for c in QUESTION_MARK_COLS if c not in df.columns]
    if qm_absent:
        print(f"[info] columns in the '?' normalisation list but absent from the CSV: {qm_absent}")
    if qm_present:
        df[qm_present] = df[qm_present].replace("?", np.nan)

    if TARGET not in df.columns:
        raise ValueError(f"The target column '{TARGET}' is missing from the CSV.")
    df[TARGET] = df[TARGET].map(TARGET_MAP)
    if df[TARGET].isna().any():
        raise ValueError(f"Column {TARGET} contains values outside {TARGET_MAP}.")
    df[TARGET] = df[TARGET].astype("int64")

    return df


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separates categorical from numeric columns.

    In pandas 3 text columns moved to the `str` dtype; `include='object'` still
    catches them for backward compatibility but emits a Pandas4Warning.
    `include=['object', 'str']` returns the same column set without the warning
    (verified).
    """
    feature_df = df.drop(columns=[TARGET])
    cat_cols = feature_df.select_dtypes(include=["object", "str"]).columns.tolist()
    num_cols = [c for c in feature_df.columns if c not in cat_cols]
    return cat_cols, num_cols


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def build_pipeline(cat_cols: list[str], scale_pos_weight: float) -> Pipeline:
    """preprocessor + model -> a single Pipeline.

    `handle_unknown='use_encoded_value', unknown_value=-1` on the OrdinalEncoder
    was NOT in the original. In a notebook that did not matter (the encoder only
    ever saw the training data), but it is unacceptable for an API: whenever a
    user sent a category unseen in training, the pipeline crashed with a
    ValueError. Now an unknown category is encoded to -1; the Phase 2 guardrail
    additionally reports the situation to the user via
    `out_of_distribution_warnings`.
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", cat_pipeline, cat_cols)],
        remainder="passthrough",
    )

    # We turn the ColumnTransformer's output into a DataFrame.
    #
    # Why: with the default numpy output, LightGBM's sklearn wrapper invents fake
    # column names such as `Column_0..Column_33` and writes them into
    # `feature_names_in_`. Every subsequent `predict` call then makes sklearn emit
    # "X does not have valid feature names, but LGBMClassifier was fitted with
    # feature names" — under FastAPI that means log noise on every request. SHAP
    # would also see those meaningless `Column_i` labels as the feature names.
    #
    # The `set_output` setting lives on the estimator and travels with the pickle,
    # so the serving side does not need a global `sklearn.set_config` call. It has
    # NO effect on the predictions: models trained with numpy and pandas output
    # produce identical predict_proba results (maximum absolute difference 0.0) —
    # verified.
    preprocessor.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGBM_PARAMS)),
        ]
    )


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def _py(value):
    """Converts numpy/pandas scalars into JSON-native types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    return value


def _require_finite(value, col: str, what: str) -> None:
    """Prevents NaN/Infinity from being written into metadata.json.

    Why `raise` rather than "silently write 0":
    by default `json.dumps` writes NaN as a BARE `NaN` token. That is NOT valid
    JSON: JavaScript's `JSON.parse` (i.e. the frontend's `/model-info` call)
    rejects the file outright. Python's `json.load` is lenient and swallows the
    problem — so the failure only surfaces in the frontend, at the latest possible
    moment.

    On the categorical side an empty mode already raised a `ValueError`; on the
    numeric side there was no check at all. This closes that asymmetry. Rather
    than inventing a default we stop the training: an entirely empty numeric
    column is a data error and must not be hidden by the artifact.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        raise ValueError(
            f"The {what} for column '{col}' is not a finite number ({value!r}). "
            "The column is most likely entirely empty in the training set. "
            "metadata.json would not be valid JSON; training stopped."
        )


def build_defaults(X_train: pd.DataFrame, cat_cols: list[str]) -> dict:
    """The defaults used to fill fields the API request did not supply.

    Numeric -> MEDIAN, categorical -> MODE.
    The original notebook applied `mode()` to every column; on a continuous
    variable such as `total_claim_amount` the mode is statistically meaningless
    (usually a random single observation). The median is both more representative
    and robust to outliers.

    The values are computed from X_train ONLY — test data never leaks into the
    defaults.

    On integer columns the median can come out as .5 (when n is even); we round
    the value to the column's training dtype so that a row reloaded from JSON has
    exactly the same types as the training data ("no type loss" requirement).
    """
    defaults: dict[str, dict] = {}

    for col in X_train.columns:
        series = X_train[col]
        dtype_name = str(series.dtype)

        if col in cat_cols:
            mode = series.mode(dropna=True)
            if mode.empty:
                raise ValueError(f"The mode for '{col}' could not be computed (all values NaN).")
            value = str(mode.iloc[0])
            strategy = "mode"
            dtype_name = "str"
        else:
            median = float(series.median())
            # On an entirely empty numeric column `median()` returns NaN, which
            # makes metadata.json invalid JSON (see `_require_finite`).
            _require_finite(median, col, "median")
            if pd.api.types.is_integer_dtype(series):
                value = round(median)
            else:
                value = float(median)
            strategy = "median"

        defaults[col] = {"value": _py(value), "dtype": dtype_name, "strategy": strategy}

    # Safety confirmation: the PII columns were dropped and cannot leak via defaults.
    leaked = [c for c in ("policy_number", "insured_zip", "incident_location") if c in defaults]
    if leaked:
        raise ValueError(f"A PII field leaked into defaults: {leaked}")

    return defaults


def build_training_ranges(
    X_train: pd.DataFrame,
    fitted_pipeline: Pipeline,
    cat_cols: list[str],
    num_cols: list[str],
) -> dict:
    """The "what did we see in training" record the Phase 2 guardrail relies on.

    If this is not captured at training time it cannot be recovered later (the CSV
    may be lost, the split may change), so it is written into the metadata.

    - Numeric: the min/max in X_train. Anything outside that is extrapolation.
    - Categorical: the fitted OrdinalEncoder's `categories_` list. Deliberately
      NOT the uniques in the raw X_train: the encoder is fitted AFTER imputation,
      so `categories_` is exactly the set of values that map to a real code.
      Anything not in that list goes down the -1 (unknown) path, which means the
      guardrail and the encoder report exactly the same truth.
    """
    ranges: dict[str, dict] = {}

    for col in num_cols:
        series = X_train[col]
        col_min = _py(series.min())
        col_max = _py(series.max())
        # On an empty numeric column min/max come out as NaN -> invalid JSON (see
        # `_require_finite`). A NaN-bounded range would also leave the Phase 2
        # guardrail silently inoperative: `x < NaN` is always False, so no value
        # would ever count as OOD.
        _require_finite(col_min, col, "min")
        _require_finite(col_max, col, "max")
        ranges[col] = {
            "type": "numeric",
            "dtype": str(series.dtype),
            "min": col_min,
            "max": col_max,
        }

    encoder = fitted_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps[
        "encoder"
    ]
    encoder_cols = list(
        fitted_pipeline.named_steps["preprocessor"].transformers_[0][2]
    )
    if encoder_cols != cat_cols:
        raise ValueError("The categorical column order in the ColumnTransformer is not as expected.")

    for col, categories in zip(encoder_cols, encoder.categories_):
        ranges[col] = {
            "type": "categorical",
            "dtype": "str",
            "categories": sorted(str(c) for c in categories),
        }

    missing = set(X_train.columns) - set(ranges)
    if missing:
        raise ValueError(f"Columns missing from training_ranges: {sorted(missing)}")

    return ranges


def clean_transformed_names(preprocessor: ColumnTransformer) -> list[str]:
    """'cat__incident_severity' / 'remainder__witnesses' -> 'incident_severity'.

    Required so the SHAP output is readable in the frontend.
    """
    return [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 — pipeline consolidation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    csv_path: Path = args.data.resolve()
    output_dir: Path = args.output_dir.resolve()
    if not csv_path.exists():
        print(f"[error] data file not found: {csv_path}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading data: {csv_path}")
    df = load_raw_frame(csv_path)
    cat_cols, num_cols = split_columns(df)
    print(f"      rows={len(df)}  categorical={len(cat_cols)}  numeric={len(num_cols)}")

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # An early class check. It closes two separate hidden failure points:
    #   - `train_test_split(stratify=y)`: raises a hard-to-read ValueError on
    #     single-class data, or data with only one sample of a class.
    #   - `scale_pos_weight`: division by zero when there is no positive class.
    # Both would only blow up halfway through the flow; here we stop early with a
    # single error that says what happened.
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            f"The target '{TARGET}' has a single class: {class_counts.to_dict()}. "
            "Binary classification requires both classes "
            f"(expected labels: {sorted(TARGET_MAP.values())})."
        )
    min_class_count = int(class_counts.min())
    # A stratified split needs at least 2 samples per class so it can put one of
    # each class into both train and test.
    if min_class_count < 2:
        raise ValueError(
            f"The class distribution of '{TARGET}' is insufficient for a stratified "
            f"split: {class_counts.to_dict()}. At least 2 samples per class are required."
        )

    # The split happens BEFORE feature engineering and BEFORE fitting the
    # pipeline. Imputer/encoder statistics are learned from X_train only (no leak).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    n_pos_train = int((y_train == 1).sum())
    if n_pos_train == 0:
        raise ValueError(
            "There is no positive sample in the training set; scale_pos_weight cannot "
            f"be computed. With test_size={TEST_SIZE} the split may have thrown a very "
            "small minority class entirely onto the test side."
        )
    scale_pos_weight = float((y_train == 0).sum() / n_pos_train)
    print(f"[2/6] Split: train={len(X_train)} test={len(X_test)}  scale_pos_weight={scale_pos_weight:.4f}")

    print("[3/6] Fitting the pipeline on RAW X_train...")
    pipeline = build_pipeline(cat_cols, scale_pos_weight)
    pipeline.fit(X_train, y_train)

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    train_pr_auc = float(average_precision_score(y_train, train_proba))
    test_pr_auc = float(average_precision_score(y_test, test_proba))

    print(f"[4/6] Train PR-AUC: {train_pr_auc:.4f}")
    print(f"      Test  PR-AUC: {test_pr_auc:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, pipeline.predict(X_test), digits=3))

    print("[5/6] Building metadata...")
    preprocessor = pipeline.named_steps["preprocessor"]
    pipeline_input_order = list(X_train.columns)
    transformed_display_names = clean_transformed_names(preprocessor)

    # The feature influence measurement is computed BEFORE the fairness section:
    # the fairness declaration builds on that measurement (declared and measured
    # are separate fields).
    feature_influence = build_feature_influence(pipeline, transformed_display_names)
    n_dead = feature_influence["summary"]["n_without_influence"]
    print(
        f"      feature influence: {feature_influence['summary']['n_with_influence']} "
        f"used / {n_dead} unused (split=0)"
    )
    if n_dead:
        print(f"      unused: {', '.join(feature_influence['summary']['features_without_influence'])}")
    metadata = {
        "model_name": "insurance-fraud-detection",
        "model_version": "1.0.0",
        "algorithm": "LightGBM (LGBMClassifier)",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "source_file": csv_path.name,
            "n_rows": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        },
        "metrics": {
            "train_pr_auc": train_pr_auc,
            "test_pr_auc": test_pr_auc,
            "metric_name": "average_precision_score (PR-AUC)",
        },
        "feature_list": {
            # NAMING WARNING: this list used to be called `raw_input_order`, but
            # its contents are NOT raw CSV columns. The raw CSV has 39 columns and
            # includes `incident_date` / `policy_bind_date` / the PII columns; the
            # 34 columns here are the schema the pipeline actually expects AFTER
            # the `load_raw_frame()` cleaning. Since Phase 1 relies on this list,
            # the name and the contents now agree.
            "pipeline_input_order": pipeline_input_order,
            "pipeline_input_order_description": (
                "The column names and order of the DataFrame to be handed to "
                "pipeline.predict_proba(). This is NOT the raw CSV schema: it is the "
                "schema AFTER the preprocessing_contract."
                "caller_must_apply_before_predict steps have been applied."
            ),
            "categorical_features": cat_cols,
            "numeric_features": num_cols,
            # The column order in the ColumnTransformer output (for the SHAP mapping).
            "transformed_order": list(preprocessor.get_feature_names_out()),
            "transformed_display_names": transformed_display_names,
            "dropped_columns": DROP_COLS,
            "derived_features": dict(DATE_YEAR_FEATURES),
            "target": TARGET,
        },
        "feature_influence": feature_influence,
        "preprocessing_contract": build_preprocessing_contract(pipeline_input_order),
        "fairness": build_fairness_section(cat_cols, num_cols, feature_influence),
        "defaults": build_defaults(X_train, cat_cols),
        "training_ranges": build_training_ranges(X_train, pipeline, cat_cols, num_cols),
        "model_params": {
            **{k: _py(v) for k, v in LGBM_PARAMS.items()},
            "scale_pos_weight": scale_pos_weight,
        },
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        },
    }

    print("[6/6] Writing artifacts...")
    pipeline_path = output_dir / "pipeline.pkl"
    metadata_path = output_dir / "metadata.json"
    joblib.dump(pipeline, pipeline_path)

    # `allow_nan=False`: the last line of defence. The per-column checks
    # (`_require_finite`) give targeted, readable errors; this one blocks the write
    # entirely if a NaN/Infinity survives ANYWHERE in the metadata. The default
    # `allow_nan=True` writes them as bare `NaN`/`Infinity` tokens — Python can
    # read them back, but that IS NOT VALID JSON and the frontend's `JSON.parse`
    # rejects the file. Let the error surface here, not in the browser.
    serialized = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
    metadata_path.write_text(serialized, encoding="utf-8")

    print(f"      {pipeline_path}  ({pipeline_path.stat().st_size / 1024:.1f} KB)")
    print(f"      {metadata_path}  ({metadata_path.stat().st_size / 1024:.1f} KB)")
    print("\nDone. pipeline.pkl alone is sufficient as the model artifact; "
          "preprocessor.pkl / best_model_lgb.pkl / defaults.pkl are no longer needed.")
    print("But the pipeline does NOT accept a raw CSV row: the caller must first "
          "apply the metadata.json -> preprocessing_contract steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
