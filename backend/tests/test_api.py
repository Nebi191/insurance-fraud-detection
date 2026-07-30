"""Phase 1 API tests.

TEST PHILOSOPHY
---------------
This file tries to do more than ask "did the endpoint return 200". Each test
closes a concrete class of SILENT failure:

  * The schema and the artifact can drift apart -> Literal/metadata sync test
  * If `"?"` is not converted to NaN, the encoder encodes it to -1 and the
    imputer never runs; the probability MAY NOT CHANGE, so an end-to-end test
    cannot catch it                              -> a test at the encoder level
  * If the wrong SHAP class axis is chosen, every sign flips while the response
    still looks "valid"                          -> the additivity test
  * If the response schema widens, PII/hyperparameters can leak
                                                 -> allowlist + PII tests
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, get_args

import annotated_types
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic.fields import FieldInfo

from app import main as main_module
from app import model as model_module
from app.main import (
    ALLOWED_ORIGINS_ENV,
    DEFAULT_ALLOWED_ORIGINS,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    MAX_REQUEST_BODY_BYTES,
    RATE_LIMIT_PER_MINUTE_ENV,
    CorsConfigurationError,
    RateLimitConfigurationError,
    _client_key,
    _FixedWindowRateLimiter,
    app,
    create_app,
    get_allowed_origins,
    get_rate_limit_per_minute,
    model_info,
)
from app.model import (
    METADATA_PATH,
    MODELS_DIR,
    PIPELINE_PATH,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    SHAP_LOCK_TIMEOUT_ENV,
    ArtifactError,
    ModelBundle,
    ShapLockConfigurationError,
    ShapLockTimeoutError,
    _positive_class_index,
    classify_risk,
    get_shap_lock_timeout_seconds,
)
from app.schemas import PREDICT_REQUEST_EXAMPLE, PredictRequest

# `train_pipeline.py` never lets these columns near the model. Their names are
# kept in one place so the leakage tests are updated from a single point.
PII_COLUMNS = ("policy_number", "insured_zip", "incident_location")

# Metadata keys that must NEVER appear in the `/model-info` response.
FORBIDDEN_MODEL_INFO_KEYS = ("preprocessing_contract", "model_params", "source_file")


# --------------------------------------------------------------------------- #
# 1) Artifact loading
# --------------------------------------------------------------------------- #


def test_artifacts_actually_load(bundle: ModelBundle) -> None:
    """pipeline.pkl and metadata.json really loaded and agree with each other."""
    assert bundle.pipeline is not None
    assert bundle.explainer is not None
    assert len(bundle.input_order) == 34
    assert set(bundle.defaults) == set(bundle.input_order)
    # If the verifications inside `ModelBundle.load()` (contract, feature
    # alignment, SHAP additivity) did not raise, we could only have got this far.


def test_pipeline_is_a_single_sklearn_pipeline(bundle: ModelBundle) -> None:
    """The single-Pipeline rule: preprocessing is an inseparable part of the model."""
    assert list(bundle.pipeline.named_steps) == ["preprocessor", "model"]


def test_artifact_path_is_derived_from_the_package_not_from_input(bundle: ModelBundle) -> None:
    """K1: the artifact path is fixed and derived from the package location.

    Loading a pickle is equivalent to executing code; letting the path come from a
    request or the environment would mean arbitrary code execution.
    """
    assert MODELS_DIR.name == "models"
    assert MODELS_DIR.parent.name == "backend"
    assert PIPELINE_PATH.parent == MODELS_DIR
    assert METADATA_PATH.parent == MODELS_DIR


def test_missing_artifact_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """With no artifact, `ModelBundle.load()` raises a clear error."""
    monkeypatch.setattr(model_module, "PIPELINE_PATH", tmp_path / "missing.pkl")
    with pytest.raises(model_module.ArtifactError, match="not found"):
        ModelBundle.load()


def test_app_refuses_to_start_without_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """K1 fail-fast: with no metadata the application does NOT start at all.

    A service that is "up but returns 500 to every request" is more dangerous than
    one that never starts, because it shows a green healthcheck.
    """
    monkeypatch.setattr(model_module, "METADATA_PATH", tmp_path / "missing.json")
    with pytest.raises(model_module.ArtifactError), TestClient(app):
        pass  # pragma: no cover - unreachable


# --------------------------------------------------------------------------- #
# 2) /health
# --------------------------------------------------------------------------- #


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


# --------------------------------------------------------------------------- #
# 3) /model-info — allowlist and leakage
# --------------------------------------------------------------------------- #


def test_model_info_returns_model_card_fields(client: TestClient, metadata: dict[str, Any]) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()

    assert body["model_version"] == metadata["model_version"]
    assert body["metrics"]["test_pr_auc"] == pytest.approx(metadata["metrics"]["test_pr_auc"])
    assert body["feature_list"]["pipeline_input_order"] == metadata["feature_list"]["pipeline_input_order"]
    assert body["trained_at"] == metadata["trained_at"]
    # K6: the thresholds and the calibration warning are published.
    assert body["risk_thresholds"]["low_below"] == RISK_THRESHOLD_MEDIUM
    assert body["risk_thresholds"]["high_at_or_above"] == RISK_THRESHOLD_HIGH
    assert body["probability_calibration"]["calibrated"] is False
    assert "not calibrated" in body["probability_calibration"]["warning"].lower()


def test_model_info_does_not_leak_internal_keys(client: TestClient) -> None:
    """preprocessing_contract / model_params / source_file must NOT be in the response.

    We look not only at the top-level keys but at the ENTIRE serialised text, so a
    leak nested somewhere deeper is caught too.
    """
    response = client.get("/model-info")
    body = response.json()

    assert "preprocessing_contract" not in body
    assert "model_params" not in body
    assert "source_file" not in body["dataset"]

    raw = response.text
    for key in FORBIDDEN_MODEL_INFO_KEYS:
        assert key not in raw, f"'{key}' leaked into the /model-info response"

    # Hyperparameter values must not leak either (model_params was removed, but
    # let us also verify the values do not surface indirectly through another field).
    assert "n_estimators" not in raw
    assert "learning_rate" not in raw


def test_model_info_pii_appears_only_as_dropped_column_declaration(client: TestClient) -> None:
    """PII column NAMES may appear only in the "we dropped these" declaration.

    NOTE — there is a tension in the contract, resolved deliberately this way:
    K9 puts `feature_list.dropped_columns` on the allowlist, but that list by
    definition CONTAINS the names `policy_number` / `insured_zip` /
    `incident_location`. That is not a leak — it is not a PII *value* but a
    statement that "these columns never reached the model", and it is positive
    information for a model card. So the test verifies this: those names appear
    nowhere OUTSIDE dropped_columns.
    """
    body = client.get("/model-info").json()

    dropped = body["feature_list"].pop("dropped_columns")
    assert set(PII_COLUMNS).issubset(set(dropped)), "the PII columns must be in the drop list"

    remainder = json.dumps(body, ensure_ascii=False)
    for column in PII_COLUMNS:
        assert column not in remainder, f"'{column}' also appears outside dropped_columns"

    # Because the PII columns never reached the model, they cannot produce a
    # default or a range either.
    for column in PII_COLUMNS:
        assert column not in body["defaults"]
        assert column not in body["training_ranges"]
        assert column not in body["feature_list"]["pipeline_input_order"]


def _bounds_via_model_fields(field_name: str) -> tuple[float, float] | None:
    """Independently derives one field's (ge, le) bound straight from
    `PredictRequest.model_fields`, walking `annotated_types.Ge`/`Le` metadata —
    NOT via `model_json_schema()`, which is what `app.schemas.
    physical_ranges_from_predict_request` itself reads.

    This is deliberately a SECOND, DIFFERENT introspection path. The production
    code and this test therefore cannot agree by construction (e.g. both reading
    the same cached schema dict); they only agree if `/model-info`'s
    `physical_ranges` genuinely reflects the field's real `Field(ge=..., le=...)`
    declaration. Returns `None` for a field with no numeric bound (categorical
    `Literal` fields).
    """
    field = PredictRequest.model_fields[field_name]
    for arg in get_args(field.annotation) or (field.annotation,):
        if arg is type(None):
            continue
        inner_args = get_args(arg)
        if not inner_args:
            continue
        for meta in inner_args[1:]:
            metadata = meta.metadata if isinstance(meta, FieldInfo) else [meta]
            ge = next((m.ge for m in metadata if isinstance(m, annotated_types.Ge)), None)
            le = next((m.le for m in metadata if isinstance(m, annotated_types.Le)), None)
            if ge is not None and le is not None:
                return float(ge), float(le)
    return None


def test_model_info_physical_ranges_match_pydantic_bounds_by_introspection(
    client: TestClient,
) -> None:
    """`physical_ranges` must be EXACTLY `PredictRequest`'s own Pydantic bounds.

    This is the single-source-of-truth guarantee: the values are compared
    against bounds DERIVED FROM THE MODEL (via a second, independent
    introspection path — `_bounds_via_model_fields` above), never against a
    hand-written expected dict. If a `Field(ge=..., le=...)` bound in
    `schemas.py` ever changes, this test changes its expectation with it
    automatically; a hand-copied dict could not do that.
    """
    body = client.get("/model-info").json()
    physical_ranges = body["physical_ranges"]

    expected: dict[str, tuple[float, float]] = {}
    for name, field in PredictRequest.model_fields.items():
        bounds = _bounds_via_model_fields(name)
        if bounds is None:
            continue  # categorical field — no physical bound to publish.
        key = field.alias if field.alias else name
        expected[key] = bounds

    assert set(physical_ranges) == set(expected), (
        "the set of fields published in physical_ranges disagrees with the "
        "numeric fields actually declared in PredictRequest"
    )
    for key, (low, high) in expected.items():
        assert physical_ranges[key] == {"min": low, "max": high}, (
            f"'{key}': /model-info published {physical_ranges[key]}, "
            f"but PredictRequest declares (min={low}, max={high})"
        )

    # The two hyphenated aliases must come through under the EXACT same spelling
    # `training_ranges` already uses (see `schemas.py`'s
    # `physical_ranges_from_predict_request` docstring) — a client must be able
    # to look a field up in both dicts with one key, no translation table.
    assert "capital-gains" in physical_ranges
    assert "capital-loss" in physical_ranges
    assert "capital_gains" not in physical_ranges
    assert "capital_loss" not in physical_ranges
    assert set(physical_ranges["capital-gains"]) == {"min", "max"}


# --------------------------------------------------------------------------- #
# 4) /predict — happy path
# --------------------------------------------------------------------------- #


def test_predict_with_claude_md_example(client: TestClient) -> None:
    """The example request from CLAUDE.md works EXACTLY as written."""
    response = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
    assert response.status_code == 200, response.text

    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["risk_level"] == classify_risk(body["fraud_probability"])


def test_predict_response_has_exactly_the_contract_fields(client: TestClient) -> None:
    """The 4 fields of the contract, no more (keep the leak surface narrow)."""
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert set(body) == {
        "fraud_probability",
        "risk_level",
        "shap_values",
        "out_of_distribution_warnings",
    }


def test_predict_with_empty_body_uses_defaults(client: TestClient) -> None:
    """It works with `{}` too: all 34 fields come from metadata.defaults."""
    response = client.post("/predict", json={})
    assert response.status_code == 200, response.text
    assert 0.0 <= response.json()["fraud_probability"] <= 1.0


def test_predict_is_deterministic(client: TestClient) -> None:
    """Same input -> same probability (bit for bit)."""
    first = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    second = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert first == second


def test_out_of_distribution_warnings_fire_end_to_end(client: TestClient) -> None:
    """Phase 2 completion criterion: an out-of-training value warns through `/predict`.

    In Phase 1 this test asserted "always empty" and it was deliberately noted
    that it would be updated in Phase 2. The updated version pins down both ends
    of the contract: a request INSIDE the range must not warn, one OUTSIDE must.
    """
    # The CLAUDE.md example is entirely inside the training range.
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert body["out_of_distribution_warnings"] == []

    # witnesses is 0-3 in training, age is 20-64. Both are physically possible, so
    # Pydantic accepts them — but the model has never seen them.
    body = client.post("/predict", json={"witnesses": 9, "age": 110}).json()
    assert set(body["out_of_distribution_warnings"]) == {"witnesses", "age"}
    # A warning does NOT block the prediction: the score still comes back.
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}


# --------------------------------------------------------------------------- #
# 5) /predict — validation rejections (422)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"incident_severity": "Catastrophic"}, "invalid Literal"),
        ({"policy_state": "TX"}, "category not seen in training"),
        ({"collision_type": "??"}, "unknown marker other than '?'"),
        ({"witnesses": 999}, "numeric upper bound"),
        ({"witnesses": -1}, "numeric lower bound"),
        ({"age": 5}, "physical lower bound"),
        ({"auto_year": 3000}, "physical upper bound"),
        ({"capital-gains": -1}, "gains cannot be negative"),
        ({"capital-loss": 100}, "a loss cannot be positive in this dataset"),
        ({"total_claim_amount": -5}, "an amount cannot be negative"),
        ({"number_of_vehicles_involved": 0}, "at least 1 vehicle"),
        ({"incident_hour_of_the_day": 24}, "a day has hours 0-23"),
        ({"unknown_field": 1}, "extra='forbid'"),
        ({"capital_gains": 1}, "underscore spelling — the alias must be 'capital-gains'"),
        ({"incident_date": "2015-01-25"}, "raw dates are not on the API surface (K2)"),
        ({"policy_bind_date": "1990-01-01"}, "raw dates are not on the API surface (K2)"),
        ({"policy_number": 521585}, "a PII field is not accepted"),
        ({"insured_zip": 466132}, "a PII field is not accepted"),
        ({"witnesses": "two"}, "type error"),
    ],
)
def test_predict_rejects_invalid_input(client: TestClient, payload: dict, reason: str) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"{reason} should have been rejected: {response.text}"


def test_predict_accepts_question_mark(client: TestClient) -> None:
    """'?' is a valid value on three fields and returns a 200."""
    response = client.post(
        "/predict",
        json={"collision_type": "?", "property_damage": "?", "police_report_available": "?"},
    )
    assert response.status_code == 200, response.text
    assert 0.0 <= response.json()["fraud_probability"] <= 1.0


# --------------------------------------------------------------------------- #
# 6) Are the preprocessing_contract steps GENUINELY applied?
# --------------------------------------------------------------------------- #


def test_question_mark_becomes_nan_not_an_unknown_category(bundle: ModelBundle) -> None:
    """The REAL test of the `question_mark_to_nan` step.

    WHY AT THE ENCODER LEVEL RATHER THAN END TO END:
    In this model the `collision_type` / `property_damage` /
    `police_report_available` features have a split count of ZERO (LightGBM never
    uses them). So even if "?" were wrongly encoded to -1, `fraud_probability`
    WOULD NOT CHANGE and an end-to-end test could never catch the bug. The
    contract violation is silent.

    So we look directly at the encoded value the pipeline sees: with correct
    behaviour it is the code of the imputer's mode value, with incorrect behaviour
    it is -1.
    """
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    cat_step = preprocessor.named_transformers_["cat"]
    encoder = cat_step.named_steps["encoder"]
    imputer = cat_step.named_steps["imputer"]
    cat_columns = list(preprocessor.transformers_[0][2])

    for column in bundle.question_mark_columns:
        frame = bundle.prepare_row({column: "?"})
        assert frame[column].isna().all(), f"'?' was not converted to NaN for '{column}'"

        index = cat_columns.index(column)
        expected_category = imputer.statistics_[index]
        expected_code = float(list(encoder.categories_[index]).index(expected_category))

        encoded = float(preprocessor.transform(frame)[f"cat__{column}"].iloc[0])
        assert encoded != -1.0, f"'{column}': '?' was encoded to -1 as an unknown category"
        assert encoded == expected_code, f"'{column}': the imputer never ran"


def test_missing_field_falls_back_to_metadata_default(bundle: ModelBundle) -> None:
    """K3: a field that was not supplied is filled with the `metadata.defaults` value."""
    frame = bundle.prepare_row({})
    for column in bundle.input_order:
        assert frame[column].iloc[0] == bundle.defaults[column]


def test_prepare_row_applies_order_columns_step(bundle: ModelBundle) -> None:
    """The `order_columns` step: column names and order match the contract exactly."""
    # The keys are supplied in reverse order on purpose.
    payload = {column: bundle.defaults[column] for column in reversed(bundle.input_order)}
    frame = bundle.prepare_row(payload)
    assert list(frame.columns) == bundle.input_order


def test_explicit_null_is_treated_as_missing(client: TestClient) -> None:
    """Sending `null` gives the same result as not sending the field at all."""
    explicit = client.post("/predict", json={"witnesses": None, "age": None}).json()
    omitted = client.post("/predict", json={}).json()
    assert explicit == omitted


# --------------------------------------------------------------------------- #
# 7) SHAP output (K7)
# --------------------------------------------------------------------------- #


def test_shap_output_shape_and_naming(client: TestClient, bundle: ModelBundle) -> None:
    """34 items, sorted by descending abs(value), no raw prefix, fields per the contract."""
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    shap_values = body["shap_values"]

    assert len(shap_values) == 34
    assert len(shap_values) == len(bundle.input_order)

    for item in shap_values:
        assert set(item) == {"feature", "value", "base_value"}
        assert not item["feature"].startswith("cat__")
        assert not item["feature"].startswith("remainder__")
        assert math.isfinite(item["value"])

    # Every feature is returned exactly once.
    assert sorted(item["feature"] for item in shap_values) == sorted(bundle.input_order)

    magnitudes = [abs(item["value"]) for item in shap_values]
    assert magnitudes == sorted(magnitudes, reverse=True), "not sorted by descending abs(value)"

    base_values = {item["base_value"] for item in shap_values}
    assert len(base_values) == 1, "base_value must be identical in every item"


def test_shap_additivity_proves_correct_class_axis(client: TestClient) -> None:
    """SHAP's fundamental identity: sum(value) + base_value = raw margin (log-odds).

    This test's real job is to prove the SHAPE ASSUMPTION: had the wrong class axis
    or the wrong row been selected, the identity would break. `shap 0.52 +
    lightgbm 4.6` returns a (1, 34) ndarray for a binary model, but if a version
    change starts returning a list per class, this test goes red.
    """
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()

    shap_values = body["shap_values"]
    margin = sum(item["value"] for item in shap_values) + shap_values[0]["base_value"]
    reconstructed = 1.0 / (1.0 + math.exp(-margin))

    assert reconstructed == pytest.approx(body["fraud_probability"], abs=1e-9)


def test_shap_values_stay_aligned_with_the_booster_contributions(bundle: ModelBundle) -> None:
    """SHAP contributions must stay aligned BY POSITION with the LightGBM booster's
    `pred_contrib` output.

    WHY THIS TEST EXISTS (the answer to reviewer MINOR-1):
    The old "direction test" was tautological — it followed mathematically from
    additivity and could never go red. More importantly, additivity DOES NOT PROVE
    that feature names are mapped correctly: shuffle the SHAP values among
    themselves and the sum stays the same, so the additivity test keeps passing.

    THE LIMIT OF THIS REFERENCE — IT IS NOT AN INDEPENDENT ORACLE:
    In this configuration (LightGBM + objective='binary'), `shap.TreeExplainer`
    does not do the computation itself; inside `shap/explainers/_tree.py` it calls
    `original_model.predict(X, pred_contrib=True)` and DELEGATES straight to the
    booster. So the two sides being compared are fed from the same numerical
    source, and this test DOES NOT VERIFY SHAP's mathematics — it must not claim
    to.

    What it does verify, then: our own mapping layer. The `zip` that pairs the
    `values` array with `transformed_display_names`, the sort by `abs`, and the
    positive-class axis selection — any of these can tear the contributions away
    from the names while the result still looks "valid". Mutations it kills:
    shifted/shuffled names, an off-by-one `zip`, the wrong row selected, and a
    sort that separates the values from the names.
    """
    payload = {
        "incident_severity": "Major Damage",
        "insured_hobbies": "chess",
        "auto_year": 2010,
        "witnesses": 2,
        "capital-gains": 30000,
        "total_claim_amount": 55000,
    }

    booster = bundle.pipeline.named_steps["model"].booster_
    transformed = bundle._transform(bundle.prepare_row(payload))
    contributions = booster.predict(transformed, pred_contrib=True)

    assert contributions.shape == (1, len(bundle.display_names) + 1)

    result = bundle.predict(payload)
    by_feature = {item["feature"]: item["value"] for item in result["shap_values"]}

    # Our clean names must correspond BY POSITION to the booster's raw names.
    for position, display_name in enumerate(bundle.display_names):
        assert by_feature[display_name] == contributions[0, position], (
            f"'{display_name}' (position {position}) disagrees with the booster contribution"
        )

    base_value = result["shap_values"][0]["base_value"]
    assert base_value == contributions[0, -1]


def test_shap_feature_names_follow_the_transformed_column_order(bundle: ModelBundle) -> None:
    """The clean names must be the booster's raw feature names with the prefix stripped.

    The oracle test above pins down the values; this one pins down where the naming
    comes from. Together they close the claim "the i-th contribution belongs to the
    i-th feature" from both ends.
    """
    raw_names = list(bundle.pipeline.named_steps["model"].booster_.feature_name())
    assert [name.split("__", 1)[-1] for name in raw_names] == bundle.display_names


# --------------------------------------------------------------------------- #
# 8) PII leakage — /predict
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("payload", [PREDICT_REQUEST_EXAMPLE, {}])
def test_predict_never_leaks_pii_column_names(client: TestClient, payload: dict) -> None:
    """No PII column name appears in the RAW TEXT of the `/predict` response.

    Because `/predict` does not echo the input, a PII *value* cannot come back
    anyway; this test raises the alarm if the schema is ever widened (e.g. if an
    "echo of the request" is added).
    """
    raw = client.post("/predict", json=payload).text
    for column in PII_COLUMNS:
        assert column not in raw
    # The other dropped columns have no business on the response surface either.
    assert "auto_model" not in raw
    assert "incident_date" not in raw


# --------------------------------------------------------------------------- #
# 9) Schema <-> artifact sync
# --------------------------------------------------------------------------- #


def test_request_schema_covers_every_pipeline_input(bundle: ModelBundle) -> None:
    """Does the Pydantic model cover all 34 of the pipeline's inputs?

    A missing field means that field can NEVER be supplied through the API (it
    always stays at its default) — a silent loss of functionality.
    """
    aliases = {
        field.alias or name for name, field in PredictRequest.model_fields.items()
    }
    assert aliases == set(bundle.input_order)


def test_literal_choices_match_metadata(metadata: dict[str, Any]) -> None:
    """Every categorical field's `Literal` options match the training categories.

    The category lists were hand-written in `schemas.py` for readability; this test
    ties them back to the artifact. If the model is retrained and a category is
    added or removed, the test goes red — the schema cannot go stale silently.
    """
    training_ranges = metadata["training_ranges"]
    question_mark_columns = set(
        next(
            step
            for step in metadata["preprocessing_contract"]["caller_must_apply_before_predict"]
            if step["step"] == "question_mark_to_nan"
        )["columns"]
    )

    categorical = metadata["feature_list"]["categorical_features"]
    for column in categorical:
        field = PredictRequest.model_fields[column]
        # Annotation: Literal[...] | None -> unwrap the Optional first.
        literal_args = set()
        for member in get_args(field.annotation):
            literal_args.update(get_args(member))

        expected = set(training_ranges[column]["categories"])
        if column in question_mark_columns:
            expected.add("?")

        assert literal_args == expected, f"'{column}' Literal options disagree with the metadata"


def test_numeric_bounds_are_supersets_of_training_ranges(metadata: dict[str, Any]) -> None:
    """K5: the Pydantic bound must CONTAIN the training range.

    A check in both directions:
      * A value from training must not get a 422 (the bound must be a superset).
      * Except for `incident_hour_of_the_day`, the bound must NOT EQUAL the
        training range; if it did, the Phase 2 guardrail could never fire on that
        field.
    """
    # The only field whose physical bound genuinely coincides with the training range.
    naturally_bounded = {"incident_hour_of_the_day"}

    # We read the bounds from the PUBLISHED JSON Schema, not from Pydantic's
    # internal fields: this is the very contract a frontend developer sees in
    # OpenAPI. On `int | None` fields the constraints live on the non-null branch
    # of the anyOf.
    properties = PredictRequest.model_json_schema()["properties"]

    for column in metadata["feature_list"]["numeric_features"]:
        branches = [
            branch
            for branch in properties[column].get("anyOf", [properties[column]])
            if branch.get("type") != "null"
        ]
        assert len(branches) == 1, f"unexpected schema branch for '{column}'"
        branch = branches[0]

        assert "minimum" in branch and "maximum" in branch, f"no ge/le defined for '{column}'"
        low, high = float(branch["minimum"]), float(branch["maximum"])

        train_min = float(metadata["training_ranges"][column]["min"])
        train_max = float(metadata["training_ranges"][column]["max"])

        assert low <= train_min, f"the lower bound of '{column}' cuts into the training minimum"
        assert high >= train_max, f"the upper bound of '{column}' cuts into the training maximum"

        if column not in naturally_bounded:
            assert (low, high) != (train_min, train_max), (
                f"the Pydantic bound of '{column}' equals the training range — "
                "the guardrail can never fire"
            )


def test_training_extremes_are_accepted(client: TestClient, metadata: dict[str, Any]) -> None:
    """The training set's min/max values must pass through the API.

    The test above compares the bounds statically; this one proves the same claim
    with a real HTTP request.
    """
    for column in metadata["feature_list"]["numeric_features"]:
        ranges = metadata["training_ranges"][column]
        for bound in ("min", "max"):
            response = client.post("/predict", json={column: ranges[bound]})
            assert response.status_code == 200, (
                f"the training value {column}={ranges[bound]} was rejected: {response.text}"
            )


# --------------------------------------------------------------------------- #
# 10) risk_level thresholds (K6)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (RISK_THRESHOLD_MEDIUM - 1e-9, "low"),
        (RISK_THRESHOLD_MEDIUM, "medium"),
        (0.5, "medium"),
        (RISK_THRESHOLD_HIGH - 1e-9, "medium"),
        (RISK_THRESHOLD_HIGH, "high"),
        (1.0, "high"),
    ],
)
def test_risk_level_thresholds(probability: float, expected: str) -> None:
    """The thresholds are closed below / open above: low < 0.35 <= medium < 0.65 <= high."""
    assert classify_risk(probability) == expected


# --------------------------------------------------------------------------- #
# 11) CORS (K10)
# --------------------------------------------------------------------------- #


def test_cors_env_var_is_parsed_as_comma_separated_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://a.netlify.app, https://b.example , ")
    assert get_allowed_origins() == ["https://a.netlify.app", "https://b.example"]


def test_cors_default_is_the_vite_dev_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the variable is NOT SET at all, the default applies (that is not an error)."""
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    assert get_allowed_origins() == ["http://localhost:5173"]
    assert DEFAULT_ALLOWED_ORIGINS == "http://localhost:5173"


@pytest.mark.parametrize(
    "value",
    ["*", "http://localhost:5173,*", " * ", "https://*.netlify.app"],
)
def test_cors_wildcard_is_rejected_at_startup(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """K10 is now enforced on the environment-variable path too (reviewer MAJOR-2).

    `ALLOWED_ORIGINS="*"` used to produce a real wildcard: the rule only held for
    the default value and a single environment variable punched through it.
    """
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, value)
    with pytest.raises(CorsConfigurationError, match="wildcard"):
        get_allowed_origins()
    # The application must not start either — the error does not stay in the
    # configuration layer.
    with pytest.raises(CorsConfigurationError):
        create_app()


@pytest.mark.parametrize("value", ["", "   ", ",", " , , "])
def test_cors_explicitly_empty_is_rejected_at_startup(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """An empty `ALLOWED_ORIGINS` must not silently shut out every client (MINOR-5)."""
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, value)
    with pytest.raises(CorsConfigurationError, match="no valid origin"):
        get_allowed_origins()


def test_cors_wiring_reflects_allowed_origin_and_rejects_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CORS WIRING itself is under test (reviewer MAJOR-1).

    Because the `create_app()` factory reads the origins at call time, tests can
    change the environment and build a fresh application. The origin list used to
    be frozen at import time, which made this path untestable — and the reviewer's
    M21/M23/M24 mutations passed silently through an untested path.
    """
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://allowed.example,https://second.example")

    with TestClient(create_app()) as scoped:
        allowed = scoped.get("/health", headers={"Origin": "https://allowed.example"})
        assert allowed.headers.get("access-control-allow-origin") == "https://allowed.example"

        second = scoped.get("/health", headers={"Origin": "https://second.example"})
        assert second.headers.get("access-control-allow-origin") == "https://second.example"

        # A disallowed origin must not be reflected (M21: the mutation that
        # hard-codes the origin).
        denied = scoped.get("/health", headers={"Origin": "https://blocked.example"})
        assert denied.headers.get("access-control-allow-origin") is None

        # The default origin is no longer on the list, so it must not be reflected either.
        stale = scoped.get("/health", headers={"Origin": "http://localhost:5173"})
        assert stale.headers.get("access-control-allow-origin") is None


def test_cors_preflight_limits_methods_and_disables_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preflight response allows GET/POST only and keeps credentials off.

    This is the test that kills the M23 (`allow_methods=["*"]`) and M24
    (`allow_credentials=True`) mutations.
    """
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://allowed.example")

    with TestClient(create_app()) as scoped:
        preflight = scoped.options(
            "/predict",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 200

        methods = {
            m.strip().upper()
            for m in preflight.headers.get("access-control-allow-methods", "").split(",")
            if m.strip()
        }
        assert methods == {"GET", "POST"}, f"unexpected method set: {methods}"
        assert "*" not in preflight.headers.get("access-control-allow-methods", "")

        # Credentials must be OFF: if they were on, the header would come back "true".
        assert preflight.headers.get("access-control-allow-credentials") is None

        # A preflight for a forbidden method must NOT be approved.
        # Starlette returns 400 + "Disallowed CORS method" here; even if the origin
        # header is present in the response, the browser still blocks the request —
        # the status code is what decides.
        rejected = scoped.options(
            "/predict",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        assert rejected.status_code == 400
        assert "Disallowed CORS method" in rejected.text


def test_cors_headers_are_present_on_error_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Error responses such as 413/422 must carry the CORS headers too.

    Otherwise a browser client cannot read the error and only sees an opaque
    network failure — that is the concrete reason for the middleware order (CORS
    outermost).
    """
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://allowed.example")

    with TestClient(create_app()) as scoped:
        invalid = scoped.post(
            "/predict",
            json={"witnesses": 999},
            headers={"Origin": "https://allowed.example"},
        )
        assert invalid.status_code == 422
        assert invalid.headers.get("access-control-allow-origin") == "https://allowed.example"

        oversized = scoped.post(
            "/predict",
            content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
            headers={"Origin": "https://allowed.example", "Content-Type": "application/json"},
        )
        assert oversized.status_code == 413
        assert oversized.headers.get("access-control-allow-origin") == "https://allowed.example"


# --------------------------------------------------------------------------- #
# 12) The 422 body must not echo client data (reviewer MINOR-2)
# --------------------------------------------------------------------------- #


def test_validation_error_does_not_echo_submitted_values(client: TestClient) -> None:
    """The VALUE submitted in a bad request must not appear in the response.

    Pydantic's default body returns the raw value in an `input` field. The body
    sent to this API is insurance claim data; a bad request would carry it into
    logs, reverse proxies and the browser console.
    """
    canary = "SECRET-CLAIM-DATA-7f3a91"
    response = client.post(
        "/predict",
        json={"insured_hobbies": canary, "witnesses": 4242, "policy_annual_premium": -12345.6},
    )

    assert response.status_code == 422
    raw = response.text
    assert canary not in raw, "the submitted value was echoed in the response"
    assert "4242" not in raw
    assert "12345.6" not in raw

    # The `input` and `ctx` fields must be gone entirely.
    for error in response.json()["detail"]:
        assert set(error) == {"loc", "msg", "type"}


def test_validation_error_still_tells_which_field_failed(client: TestClient) -> None:
    """Sanitisation must NOT destroy diagnostic value — the frontend shows per-field errors."""
    response = client.post("/predict", json={"witnesses": 999, "age": 5})
    assert response.status_code == 422

    detail = response.json()["detail"]
    failed_fields = {error["loc"][-1] for error in detail}
    assert failed_fields == {"witnesses", "age"}

    for error in detail:
        assert error["loc"][0] == "body"
        assert error["msg"], "the error message must not be empty"
        assert error["type"], "the machine-readable error type must not be lost"


def test_unknown_field_error_names_the_field_but_not_its_value(client: TestClient) -> None:
    """On an unknown field the KEY name comes back, the VALUE does not."""
    response = client.post("/predict", json={"secret_field": "very-secret-value-991"})
    assert response.status_code == 422
    assert "secret_field" in response.text
    assert "very-secret-value-991" not in response.text


# --------------------------------------------------------------------------- #
# 13) Body size limit (reviewer MINOR-6)
# --------------------------------------------------------------------------- #


def test_body_within_limit_is_processed_normally(client: TestClient) -> None:
    """A body under the limit enters the normal flow (200 or 422)."""
    response = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
    assert response.status_code == 200


def test_oversized_body_is_rejected_with_413(client: TestClient) -> None:
    """A large body declared via `Content-Length` is rejected before being read."""
    payload = b'{"insured_hobbies":"' + b"a" * (MAX_REQUEST_BODY_BYTES + 1) + b'"}'
    response = client.post(
        "/predict", content=payload, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert "detail" in response.json()


def test_oversized_chunked_body_is_rejected_with_413(client: TestClient) -> None:
    """We must be protected when there is NO `Content-Length` (chunked transfer encoding).

    Trusting the header alone would grant an unbounded body to any client that
    simply omits it.
    """

    def chunks() -> Any:
        yield b'{"insured_hobbies":"'
        for _ in range(16):
            yield b"a" * 8192
        yield b'"}'

    response = client.post(
        "/predict", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_body_just_under_the_limit_reaches_validation(client: TestClient) -> None:
    """A body just under the limit gets a normal validation error, NOT a 413.

    Proves the limit has not slipped to the wrong side (off-by-one).
    """
    filler = "a" * (MAX_REQUEST_BODY_BYTES - 200)
    response = client.post(
        "/predict", content=json.dumps({"insured_hobbies": filler}).encode()
    )
    assert response.status_code == 422, response.status_code


# --------------------------------------------------------------------------- #
# 14) The OpenAPI surface (reviewer MINOR-3)
# --------------------------------------------------------------------------- #


def test_openapi_does_not_leak_internal_or_pii_names(client: TestClient) -> None:
    """`/openapi.json` is a public surface too and must be scanned.

    The forbidden names had been removed from the `/model-info` response but were
    leaking into the OpenAPI `description` fields through class docstrings. The
    explanations were moved into `#` comments; this test guarantees the surface
    stays clean.
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200

    raw = response.text
    for key in (*FORBIDDEN_MODEL_INFO_KEYS, "n_estimators", "learning_rate", *PII_COLUMNS):
        assert key not in raw, f"'{key}' leaked into /openapi.json"


def test_openapi_still_documents_the_public_contract(client: TestClient) -> None:
    """Cleaning up the leak must not empty out the documentation."""
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == {"/health", "/predict", "/model-info"}
    assert "capital-gains" in spec["components"]["schemas"]["PredictRequest"]["properties"]


# --------------------------------------------------------------------------- #
# 15) Concurrency and the single-transform shortcut (reviewer MINOR-4 + regression)
# --------------------------------------------------------------------------- #


def test_concurrent_predictions_are_bitwise_identical(client: TestClient) -> None:
    """50 parallel `/predict` calls must give a bit-identical result for the same input.

    The shared state (`explainer.expected_value` is reassigned on every call) is
    protected by a lock; this test catches a regression of that protection.
    """
    workers = 50

    def call() -> tuple[float, str]:
        body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
        return body["fraud_probability"], json.dumps(body["shap_values"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: call(), range(workers)))

    assert len({probability for probability, _ in results}) == 1
    assert len({shap_blob for _, shap_blob in results}) == 1


def test_split_transform_matches_full_pipeline_bitwise(bundle: ModelBundle) -> None:
    """The `predict()` shortcut gives the same result as the full `pipeline.predict_proba`.

    For performance, `predict()` runs the preprocessor once and hands the same
    matrix to both the model and the explainer. This test proves the shortcut has
    not drifted silently (it is also checked at startup by
    `_verify_transform_equivalence`).
    """
    payload = PredictRequest(**PREDICT_REQUEST_EXAMPLE).model_dump(by_alias=True)
    frame = bundle.prepare_row(payload)

    via_pipeline = bundle.pipeline.predict_proba(frame)
    via_steps = bundle.pipeline.named_steps["model"].predict_proba(bundle._transform(frame))

    assert np.array_equal(via_pipeline, via_steps)
    assert bundle.predict(payload)["fraud_probability"] == float(via_pipeline[0, 1])


# --------------------------------------------------------------------------- #
# 16) Library version drift (reviewer C1)
# --------------------------------------------------------------------------- #


def test_version_drift_emits_a_warning(
    bundle: ModelBundle, caplog: pytest.LogCaptureFixture
) -> None:
    """If the training and runtime versions diverge, a WARNING is emitted.

    We do not fail: version drift is usually harmless, and refusing to start on
    every drift would needlessly block the Phase 7 deploy. But staying silent is
    the most expensive form of "why is the model scoring differently?".
    """
    drifted = copy.deepcopy(bundle.metadata)
    drifted["library_versions"]["lightgbm"] = "0.0.1-old"
    drifted["library_versions"]["scikit-learn"] = "0.0.2-old"

    probe = ModelBundle(bundle.pipeline, drifted)
    with caplog.at_level(logging.WARNING, logger="app.model"):
        probe._warn_on_library_version_drift()

    assert "version drift" in caplog.text
    assert "lightgbm" in caplog.text
    assert "scikit-learn" in caplog.text


def test_no_warning_when_versions_match(
    bundle: ModelBundle, caplog: pytest.LogCaptureFixture
) -> None:
    """If the versions agree, the log is not polluted."""
    with caplog.at_level(logging.WARNING, logger="app.model"):
        bundle._warn_on_library_version_drift()
    assert "version drift" not in caplog.text


def test_version_drift_is_not_exposed_on_health(client: TestClient) -> None:
    """Drift information does not leak into the API contract — an operator signal, not client data."""
    body = client.get("/health").json()
    assert set(body) == {"status", "model_loaded", "model_version"}


# --------------------------------------------------------------------------- #
# 17) Feature influence data (F9) — the API side of the "flag and show" decision
# --------------------------------------------------------------------------- #


def test_feature_influence_is_in_metadata_and_covers_every_field(
    metadata: dict[str, Any],
) -> None:
    """There is a measurement for all 34 features and the keys match the API field names.

    Matching the API names is essential: if the frontend had to keep a separate
    mapping table, that table would go stale silently when the model changed.
    """
    influence = metadata["feature_influence"]
    features = influence["features"]

    assert len(features) == 34
    assert set(features) == set(metadata["feature_list"]["pipeline_input_order"])

    # The hyphenated field name must be preserved (same as the Pydantic alias).
    assert "capital-gains" in features
    assert "capital_gains" not in features

    for name, entry in features.items():
        assert set(entry) == {"split_count", "gain", "has_influence"}
        assert entry["has_influence"] == (entry["split_count"] > 0), name
        assert entry["split_count"] >= 0
        assert entry["gain"] >= 0.0


def test_feature_influence_summary_matches_the_per_feature_data(
    metadata: dict[str, Any],
) -> None:
    """The summary numbers agree with the detail — the summary is not a hand-written constant."""
    influence = metadata["feature_influence"]
    features = influence["features"]
    summary = influence["summary"]

    dead = sorted(name for name, entry in features.items() if not entry["has_influence"])

    assert summary["n_features"] == len(features)
    assert summary["n_without_influence"] == len(dead)
    assert summary["n_with_influence"] == len(features) - len(dead)
    assert summary["features_without_influence"] == dead
    assert summary["n_with_influence"] + summary["n_without_influence"] == summary["n_features"]


def test_sixteen_features_are_dead_and_insured_sex_is_one_of_them(
    metadata: dict[str, Any],
) -> None:
    """The measured fact: 16 features are dead, and `insured_sex` is one of them."""
    summary = metadata["feature_influence"]["summary"]

    assert summary["n_without_influence"] == 16
    assert summary["n_with_influence"] == 18
    assert "insured_sex" in summary["features_without_influence"]
    # The other dead features that bear directly on the fairness declaration.
    for feature in ("insured_relationship", "policy_state", "incident_state"):
        assert feature in summary["features_without_influence"]
    # The ones genuinely in use must NOT be on the dead list.
    for feature in ("insured_hobbies", "incident_severity", "age", "insured_education_level"):
        assert feature not in summary["features_without_influence"]


def test_feature_influence_matches_the_booster_exactly(
    bundle: ModelBundle, metadata: dict[str, Any]
) -> None:
    """The numbers in the metadata are IDENTICAL to those computed from the booster.

    Proof that they are not hand-written constants: we read the same number
    independently from the booster and compare.
    """
    booster = bundle.pipeline.named_steps["model"].booster_
    split_counts = booster.feature_importance(importance_type="split")
    gains = booster.feature_importance(importance_type="gain")
    features = metadata["feature_influence"]["features"]

    for position, display_name in enumerate(bundle.display_names):
        entry = features[display_name]
        assert entry["split_count"] == int(split_counts[position]), display_name
        assert entry["gain"] == pytest.approx(float(gains[position])), display_name
        assert entry["has_influence"] == (int(split_counts[position]) > 0)


def test_model_info_exposes_feature_influence(client: TestClient) -> None:
    """`/model-info` returns the section and the allowlist is intact."""
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()

    influence = body["feature_influence"]
    assert len(influence["features"]) == 34
    assert influence["summary"]["n_without_influence"] == 16
    assert influence["features"]["insured_sex"]["has_influence"] is False
    assert influence["features"]["insured_hobbies"]["has_influence"] is True
    assert "capital-gains" in influence["features"]

    # The allowlist still does not leak.
    raw = response.text
    for key in FORBIDDEN_MODEL_INFO_KEYS:
        assert key not in raw


def test_dead_features_have_exactly_zero_shap_value(client: TestClient) -> None:
    """Every feature with `has_influence: false` has a SHAP contribution of EXACTLY 0.0.

    An end-to-end verification of the `feature_influence` contract: the frontend
    renders the "this field does not affect the model" badge from this data, so we
    have to prove the badge matches reality.
    """
    influence = client.get("/model-info").json()["feature_influence"]["features"]
    shap_values = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()["shap_values"]
    by_feature = {item["feature"]: item["value"] for item in shap_values}

    dead = [name for name, entry in influence.items() if not entry["has_influence"]]
    assert len(dead) == 16

    for name in dead:
        assert by_feature[name] == 0.0, f"'{name}' is dead but its SHAP contribution is not zero"

    # The other direction: if every influential feature were zero too, the test
    # would be meaningless.
    alive = [name for name, entry in influence.items() if entry["has_influence"]]
    assert any(by_feature[name] != 0.0 for name in alive)


# --------------------------------------------------------------------------- #
# 18) Fairness: declaration separated from measurement (F10)
# --------------------------------------------------------------------------- #


def test_fairness_entries_carry_both_declaration_and_measurement(
    client: TestClient,
) -> None:
    """Every protected/proxy attribute carries both a declaration and a measurement field."""
    fairness = client.get("/model-info").json()["fairness"]
    entries = (
        fairness["protected_attributes_used_as_features"] + fairness["proxy_risk_attributes"]
    )
    assert entries

    for entry in entries:
        assert entry["used_as_model_feature"] is True, "the declaration must be preserved"
        assert isinstance(entry["split_count"], int)
        assert entry["has_influence"] == (entry["split_count"] > 0)


def test_fairness_split_counts_come_from_the_booster(
    client: TestClient, bundle: ModelBundle
) -> None:
    """The `split_count` in the fairness section is identical to the booster's number.

    Proof that it is NOT a hand-written constant. Also verified by mutation: a
    change that pins the number turns this test red.
    """
    booster = bundle.pipeline.named_steps["model"].booster_
    split_counts = booster.feature_importance(importance_type="split")
    expected = {
        name: int(split_counts[position])
        for position, name in enumerate(bundle.display_names)
    }

    fairness = client.get("/model-info").json()["fairness"]
    entries = (
        fairness["protected_attributes_used_as_features"] + fairness["proxy_risk_attributes"]
    )

    for entry in entries:
        assert entry["split_count"] == expected[entry["feature"]], entry["feature"]


def test_fairness_reports_measured_zero_influence_for_sex(client: TestClient) -> None:
    """`insured_sex` was declared as given, but its measured influence is zero."""
    fairness = client.get("/model-info").json()["fairness"]
    by_feature = {
        entry["feature"]: entry for entry in fairness["protected_attributes_used_as_features"]
    }

    sex = by_feature["insured_sex"]
    assert sex["used_as_model_feature"] is True
    assert sex["split_count"] == 0
    assert sex["has_influence"] is False

    # Age, on the other hand, really is used — so the measurement distinction works.
    assert by_feature["age"]["has_influence"] is True


# Phrases that would exculpate the model. Each may appear ONLY inside a sentence
# that also negates or qualifies it.
EXCULPATORY_PHRASES = (
    "the model is fair",
    "does not discriminate",
    "no discrimination",
    "free of bias",
    "unbiased",
)

# Markers that turn an exculpatory phrase into a disclaimer rather than a claim.
DISCLAIMER_MARKERS = ("not mean", "not a substitute", "not prove", "cannot be concluded")


def test_fairness_does_not_claim_the_model_is_fair(client: TestClient) -> None:
    """Adding a measurement must NOT be turned into "the model is fair".

    `has_influence: false` only means "the measured influence in this trained set
    of trees is zero". The audit still has not been performed and proxy features
    can carry the signal back; the model card has to say so explicitly.

    HOW THE NEGATIVE CHECK WORKS: a plain "this phrase must not appear" test does
    not survive translation — the honest text itself contains "the model is fair"
    inside the sentence "That DOES NOT MEAN the model is fair". So the check is
    made SENTENCE BY SENTENCE: an exculpatory phrase is allowed only when the same
    sentence also carries a disclaimer marker. A rewrite that drops the negation
    turns this red, which is exactly the failure worth catching.
    """
    fairness = client.get("/model-info").json()["fairness"]

    assert fairness["audit_performed"] is False
    assert fairness["status"] == "declared_not_audited"
    assert fairness["audit_metrics_computed"] == []
    assert fairness["production_requirements"], "the production requirements must not be deleted"

    text = (fairness["notes"] + " " + fairness["field_semantics"]).lower()

    # The limits of the measurement must be stated explicitly.
    assert "audit" in text
    assert "proxy" in text
    assert any(marker in text for marker in DISCLAIMER_MARKERS)

    # No sentence may make an exculpatory claim without qualifying it.
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        for phrase in EXCULPATORY_PHRASES:
            if phrase in sentence:
                assert any(marker in sentence for marker in DISCLAIMER_MARKERS), (
                    f"unqualified exculpatory claim: {sentence.strip()!r}"
                )


# --------------------------------------------------------------------------- #
# 19) The model layer, called directly
# --------------------------------------------------------------------------- #


def test_bundle_predict_matches_endpoint(client: TestClient, bundle: ModelBundle) -> None:
    """The HTTP layer performs no computation: endpoint output = bundle output."""
    direct = bundle.predict(PredictRequest(**PREDICT_REQUEST_EXAMPLE).model_dump(by_alias=True))
    via_http = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert direct["fraud_probability"] == pytest.approx(via_http["fraud_probability"], abs=0)
    assert direct["risk_level"] == via_http["risk_level"]


def test_unknown_category_never_reaches_the_model(bundle: ModelBundle) -> None:
    """The encoder's -1 path must be UNREACHABLE through the API.

    The pipeline is configured with `unknown_value=-1`, so an unknown category is
    silently encoded as -1. Because `Literal` closes that off at the API level, we
    must not see -1 on a single categorical field.
    """
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    frame = bundle.prepare_row(PredictRequest(**PREDICT_REQUEST_EXAMPLE).model_dump(by_alias=True))
    transformed = preprocessor.transform(frame)

    categorical = [c for c in transformed.columns if c.startswith("cat__")]
    codes = transformed[categorical].to_numpy(dtype=float)
    assert not np.any(codes < 0), "an unknown category (-1) appeared on a valid request"


# --------------------------------------------------------------------------- #
# 20) Verification round — closing the mutations that survived
# --------------------------------------------------------------------------- #
#
# The four tests below pin down claims that could NOT BE KILLED BY MUTATION in the
# previous round. A claim being written in prose does not make it true; each one
# counts as proven only once a test goes red when the relevant code is broken.


def test_shap_lock_is_actually_held_during_explanation(
    bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_shap_lock` must GENUINELY be held while the explainer runs.

    WHY A SEPARATE TEST WAS NEEDED (surviving mutation A-M8):
    `test_concurrent_predictions_are_bitwise_identical` CANNOT KILL the mutation
    that removes the lock, and that is not surprising — the `expected_value` shap
    reassigns on every call is always set to the SAME value, so the race produces
    no observable difference in the output. The lock therefore closes off not a
    measurable bug today but the library's BEHAVIOUR of mutating shared state;
    what it protects against is a future shap release where that value becomes
    input-dependent.

    An outcome-based test can never prove that, so we observe the protection
    directly: is the lock held while the explanation is produced? If the
    `with self._shap_lock:` line is deleted, this goes red.
    """
    original_explainer = bundle.explainer
    lock_states: list[bool] = []

    class _LockObservingExplainer:
        """Wraps the explainer; records the lock state AT THE MOMENT it is called."""

        def __call__(self, transformed: Any) -> Any:
            lock_states.append(bundle._shap_lock.locked())
            return original_explainer(transformed)

    monkeypatch.setattr(bundle, "explainer", _LockObservingExplainer())
    bundle.predict(PREDICT_REQUEST_EXAMPLE)

    assert lock_states == [True], (
        "the SHAP explanation was produced without holding the lock — the shared "
        "explainer state is exposed to concurrent requests."
    )


def test_load_runs_every_startup_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ModelBundle.load()` must call ALL FIVE startup verifications.

    WHY A SEPARATE TEST WAS NEEDED (surviving mutation):
    The mutation that deletes the CALL to `_verify_transform_equivalence()` from
    `load()` left the whole suite green. The method itself was tested, but not the
    fact that it is called — because the artifact is already consistent, skipping
    the verification makes no observable difference. Yet the protection exists
    precisely for an inconsistent artifact: remove it silently and the fail-fast
    guarantee is only on paper.
    """
    verifications = (
        "_verify_contract",
        "_verify_feature_alignment",
        "_verify_transform_equivalence",
        "_verify_shap_additivity",
        "_warn_on_library_version_drift",
    )
    called: list[str] = []

    def spy(name: str) -> Any:
        original = getattr(ModelBundle, name)

        def wrapper(self: ModelBundle) -> Any:
            called.append(name)
            return original(self)

        return wrapper

    for name in verifications:
        monkeypatch.setattr(ModelBundle, name, spy(name))

    ModelBundle.load()

    assert set(called) == set(verifications), (
        f"verification(s) skipped at startup: {sorted(set(verifications) - set(called))}"
    )
    # The contract check must come FIRST: the other verifications work on the
    # assumption that the metadata has the expected shape.
    assert called[0] == "_verify_contract"


def test_dead_features_change_neither_the_score_nor_their_own_shap_value(
    bundle: ModelBundle, metadata: dict[str, Any]
) -> None:
    """Changing a dead feature's VALUE changes neither the probability nor its SHAP contribution.

    WHY A SEPARATE TEST WAS NEEDED (auditing the fairness text for overclaiming):
    `fairness.field_semantics` claims that "a feature with split_count=0 cannot
    affect any prediction in this trained model, and its SHAP contribution is
    ALWAYS exactly 0.0". That text is shown to an insurance client through
    `/model-info`. The existing `test_dead_features_have_exactly_zero_shap_value`
    measured it on ONE input; an "always" claim is not proven by a single example.

    Here we try each dead feature's EXTREME values from training one by one and
    prove two things at once: (1) the score stays bit-identical, (2) that feature's
    own SHAP contribution is exactly 0.0. `insured_sex` and `insured_relationship`
    are in this set too — so "changing sex does not change the score" is now a
    measurement, not a declaration.

    CAUTION — THIS IS NOT A FAIRNESS AUDIT. The only thing proven is that the
    direct effect of these columns in THIS trained set of trees is zero. Proxy
    features (occupation, hobbies, geography) can carry the same signal
    indirectly, and no group-level metric was computed — `fairness.notes` already
    says so.
    """
    influence = metadata["feature_influence"]["features"]
    ranges = metadata["training_ranges"]

    baseline = bundle.predict({})
    baseline_probability = baseline["fraud_probability"]

    dead = [name for name, entry in influence.items() if not entry["has_influence"]]
    assert len(dead) == 16, "the number of dead features changed — metadata and test have drifted"

    checked = 0
    for name in dead:
        spec = ranges[name]
        if spec["type"] == "categorical":
            candidates: list[Any] = list(spec["categories"])
        else:
            # For numeric fields, both ends of the training range: the
            # no-influence claim must hold at the extremes too.
            candidates = [spec["min"], spec["max"]]

        for value in candidates:
            result = bundle.predict({name: value})
            assert result["fraud_probability"] == baseline_probability, (
                f"'{name}' is declared dead but setting it to {value!r} changed the score"
            )
            contribution = next(
                item["value"] for item in result["shap_values"] if item["feature"] == name
            )
            assert contribution == 0.0, (
                f"'{name}' is declared dead but with input {value!r} its SHAP "
                f"contribution is {contribution!r}"
            )
            checked += 1

    # Proof that the test really did work: many values across 16 features.
    assert checked > 40, f"fewer combinations were tried than expected: {checked}"

    # The other direction — changing a live feature MUST change the score,
    # otherwise the equalities above would just reflect a model that reacts to
    # nothing.
    moved = bundle.predict({"insured_hobbies": "chess"})["fraud_probability"]
    assert moved != baseline_probability


def test_model_info_projection_drops_unknown_nested_keys(bundle: ModelBundle) -> None:
    """The allowlist does not leak on NESTED keys either.

    WHY A SEPARATE TEST WAS NEEDED (the feature_influence leak test):
    The existing leak tests look for the three keys that exist in the metadata
    TODAY (`preprocessing_contract`, `model_params`, `source_file`). But the
    allowlist's real promise is forward-looking: "if train_pipeline.py adds a new
    field to the metadata tomorrow, that field does not go out until it is
    explicitly allowlisted". That promise cannot be tested by searching for keys
    that already exist.

    So we inject FAKE fields into the metadata and run the projection. If a
    sub-model's `extra="ignore"` is changed to `extra="allow"`, or if
    `ModelInfoResponse` starts reflecting the metadata as-is, this goes red.
    """
    probe_metadata = copy.deepcopy(bundle.metadata)

    # Fields NOT on the allowlist, scattered across three separate nesting levels.
    probe_metadata["_internal_note"] = "LEAK-ROOT"
    probe_metadata["feature_influence"]["_debug_dump"] = "LEAK-INFLUENCE"
    probe_metadata["feature_influence"]["features"]["age"]["_raw_tree_paths"] = "LEAK-ENTRY"
    probe_metadata["feature_influence"]["summary"]["_internal_rank"] = "LEAK-SUMMARY"
    probe_metadata["fairness"]["_reviewer_private_note"] = "LEAK-FAIRNESS"
    probe_metadata["fairness"]["protected_attributes_used_as_features"][0]["_ticket"] = (
        "LEAK-ATTRIBUTE"
    )
    probe_metadata["dataset"]["source_file"] = "C:/server/secret/path/insurance_claims.csv"
    probe_metadata["metrics"]["_internal_holdout_auc"] = "LEAK-METRIC"

    probe = ModelBundle(bundle.pipeline, probe_metadata)
    rendered = json.dumps(
        model_info(probe).model_dump(by_alias=True), ensure_ascii=False, default=str
    )

    assert "LEAK" not in rendered, "the allowlist leaked a nested key"
    assert "secret" not in rendered, "a server-side file path leaked"
    assert "source_file" not in rendered

    # The injection must genuinely have been made where the response TOUCHES —
    # otherwise the test would stay green while proving nothing.
    assert "insured_sex" in rendered
    assert "_raw_tree_paths" in json.dumps(probe_metadata["feature_influence"]["features"]["age"])


def test_model_info_projection_covers_training_ranges_and_defaults(
    bundle: ModelBundle,
) -> None:
    """Canaries are injected into the `training_ranges` and `defaults` branches too.

    Codex C-6: the test above covered four metadata branches but skipped these two.
    `TrainingRangeInfo` / `DefaultInfo` are `extra="ignore"` today — so the code is
    safe. But an uncovered branch leaves the claim "the allowlist works
    everywhere" untested: if one of them ever becomes `extra="allow"` and the
    metadata later puts a sensitive field there, the leak happens while the tests
    stay green.
    """
    probe_metadata = copy.deepcopy(bundle.metadata)

    a_range = next(iter(probe_metadata["training_ranges"]))
    a_default = next(iter(probe_metadata["defaults"]))
    probe_metadata["training_ranges"][a_range]["_raw_column_sample"] = "LEAK-RANGE"
    probe_metadata["defaults"][a_default]["_source_row_id"] = "LEAK-DEFAULT"

    probe = ModelBundle(bundle.pipeline, probe_metadata)
    rendered = json.dumps(
        model_info(probe).model_dump(by_alias=True), ensure_ascii=False, default=str
    )

    assert "LEAK" not in rendered
    # The branches really are in the response — the injection did not go into a void.
    assert a_range in json.loads(rendered)["training_ranges"]
    assert a_default in json.loads(rendered)["defaults"]


# --------------------------------------------------------------------------- #
# 21) The fail-fast gaps opened up by the Codex second opinion
# --------------------------------------------------------------------------- #


def test_body_limit_applies_to_endpoints_that_never_read_the_body(
    client: TestClient,
) -> None:
    """The body limit must apply on endpoints that never read the body (Codex C-1).

    THE GAP FOUND: the streaming arm of the limit depended on the application
    calling `receive()`. `/health` and `/model-info` never read the request body,
    so a client that sends no `Content-Length` (chunked) could stream an unbounded
    body to those endpoints. Measured: `GET /health` with a 160 KB chunked body ->
    200.

    When `Content-Length` IS present the cheap path already caught it; the gap was
    only on the chunked path — that is, exactly with a client that omits the
    header.
    """

    def chunks() -> Any:
        for _ in range(20):
            yield b"x" * 8192  # 160 KB in total, NO Content-Length

    for path in ("/health", "/model-info"):
        response = client.request("GET", path, content=chunks())
        assert response.status_code == 413, (
            f"{path} accepted a 160 KB chunked body ({response.status_code}) — "
            "the body limit is not applied on this endpoint"
        )

    # Normal requests must be unaffected: a bodyless GET still works.
    assert client.get("/health").status_code == 200
    assert client.get("/model-info").status_code == 200


def test_display_names_out_of_order_fails_fast(bundle: ModelBundle) -> None:
    """If `transformed_display_names` is PERMUTED, the application must not start (Codex C-3).

    THE GAP FOUND: `_verify_feature_alignment` only looked at the length and the
    `cat__` / `remainder__` prefix; neither says anything about the ORDER of the
    names. With metadata containing the same names in a different order the service
    started cleanly and the scores stayed correct, but `/predict` attributed every
    SHAP contribution to the WRONG feature.

    `_verify_shap_additivity` cannot catch this — the sum is unchanged, so the
    identity holds. The tests caught it, startup did not; a broken artifact could
    therefore reach production. That would be the very thing this demo is selling
    (explainability) lying silently.
    """
    broken = copy.deepcopy(bundle.metadata)
    names = broken["feature_list"]["transformed_display_names"]
    names[0], names[1] = names[1], names[0]  # only the ORDER is broken

    probe = ModelBundle(bundle.pipeline, broken)
    with pytest.raises(ArtifactError, match="column order"):
        probe._verify_feature_alignment()

    # Robustness: intact metadata must pass the same check.
    ModelBundle(bundle.pipeline, copy.deepcopy(bundle.metadata))._verify_feature_alignment()


def test_missing_positive_class_fails_fast_instead_of_guessing(
    bundle: ModelBundle,
) -> None:
    """If the positive label is missing from `classes_`, it must not be guessed (Codex C-2).

    THE GAP FOUND: the old code fell back to "the last class" when it could not
    find the label. When that assumption is wrong, both the `predict_proba` column
    and the SHAP axis shift AT THE SAME TIME — so `_verify_shap_additivity` uses
    the same wrong axis and the identity holds. The bug would slip past its own
    verification and bring up a service that was "sign-flipped but self-consistent".
    """

    class _WrongLabels:
        classes_ = np.array([0, 2])

    class _NoLabels:
        pass

    with pytest.raises(ArtifactError, match="positive class"):
        _positive_class_index(_WrongLabels())

    with pytest.raises(ArtifactError, match="classes_"):
        _positive_class_index(_NoLabels())

    # The real artifact must be unaffected: the positive class is 1 and its index
    # is found correctly.
    assert _positive_class_index(bundle.pipeline.named_steps["model"]) == 1


# --------------------------------------------------------------------------- #
# 21) Rate limiting — `/predict` only (Phase 7)
# --------------------------------------------------------------------------- #
#
# The shared, session-scoped `client` fixture is deliberately AVOIDED for the
# "exceeded" cases: `tests/conftest.py` gives that app an effectively unlimited
# `RATE_LIMIT_PER_MINUTE` so the rest of the suite is not order-dependent on
# how many `/predict` calls came before it. Each test below builds its OWN app
# via `create_app()` with a small, explicit limit instead (the same pattern
# `test_cors_wiring_reflects_allowed_origin_and_rejects_others` already uses
# for `ALLOWED_ORIGINS`).


def test_rate_limit_per_minute_env_defaults_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RATE_LIMIT_PER_MINUTE_ENV, raising=False)
    assert get_rate_limit_per_minute() == DEFAULT_RATE_LIMIT_PER_MINUTE == 30

    monkeypatch.setenv(RATE_LIMIT_PER_MINUTE_ENV, "5")
    assert get_rate_limit_per_minute() == 5


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1", "1.5"])
def test_rate_limit_per_minute_rejects_invalid_values_at_startup(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A malformed `RATE_LIMIT_PER_MINUTE` must fail fast, not silently disable itself."""
    monkeypatch.setenv(RATE_LIMIT_PER_MINUTE_ENV, value)
    with pytest.raises(RateLimitConfigurationError):
        get_rate_limit_per_minute()
    with pytest.raises(RateLimitConfigurationError):
        create_app()


def test_predict_rate_limit_returns_429_with_retry_after_once_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RATE_LIMIT_PER_MINUTE_ENV, "2")
    with TestClient(create_app()) as scoped:
        first = scoped.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
        second = scoped.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
        assert first.status_code == 200
        assert second.status_code == 200

        third = scoped.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
        assert third.status_code == 429
        retry_after = third.headers.get("retry-after")
        assert retry_after is not None
        assert int(retry_after) >= 1

        # The body must be generic: no IP, no per-client counter, nothing beyond
        # a fixed message (the same "do not leak anything beyond the documented
        # contract" rule `/predict`'s success response follows).
        assert third.json() == {"detail": "Too many requests. Please slow down and retry shortly."}


def test_health_and_model_info_are_exempt_from_the_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF's healthcheck/uptime polling must never see a 429."""
    monkeypatch.setenv(RATE_LIMIT_PER_MINUTE_ENV, "1")
    with TestClient(create_app()) as scoped:
        for _ in range(5):
            assert scoped.get("/health").status_code == 200
        for _ in range(5):
            assert scoped.get("/model-info").status_code == 200

        # Sanity check on the SAME app: with limit=1, `/predict` really does
        # trip on the second call — proving the limiter is live, not merely
        # unreachable from these two routes.
        assert scoped.post("/predict", json=PREDICT_REQUEST_EXAMPLE).status_code == 200
        assert scoped.post("/predict", json=PREDICT_REQUEST_EXAMPLE).status_code == 429


def test_client_key_uses_the_last_hop_of_x_forwarded_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documents the honest tradeoff: the LAST entry is taken, never the first.

    Taking the first entry blindly would let a client set its own key on every
    request (`X-Forwarded-For: <anything-i-like>`), which makes a rate limiter a
    no-op. Taking the last entry assumes exactly one trusted proxy hop that
    APPENDS the peer address (see `_client_key`'s docstring for the caveat).
    """

    def _request(headers: dict[str, str], client_host: str | None = "203.0.113.5") -> Any:
        from starlette.requests import Request

        scope = {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": (client_host, 12345) if client_host is not None else None,
        }
        return Request(scope)

    spoofed_first_hop = _request({"x-forwarded-for": "1.2.3.4, 10.0.0.9"})
    assert _client_key(spoofed_first_hop) == "10.0.0.9"

    no_header = _request({})
    assert _client_key(no_header) == "203.0.113.5"

    no_header_no_client = _request({}, client_host=None)
    assert _client_key(no_header_no_client) == "unknown"


def test_fixed_window_rate_limiter_sweeps_stale_keys_on_window_rollover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded memory: a key from a PAST window must be dropped, not accumulate forever."""
    fake_now = [0.0]
    monkeypatch.setattr(main_module.time, "monotonic", lambda: fake_now[0])

    limiter = _FixedWindowRateLimiter(limit=10, window_seconds=60.0)
    limiter.check("stale-client")
    assert "stale-client" in limiter._buckets

    fake_now[0] = 61.0  # one full window later
    limiter.check("fresh-client")

    assert "stale-client" not in limiter._buckets, "a key from a past window was never swept"
    assert set(limiter._buckets) == {"fresh-client"}


def test_fixed_window_rate_limiter_resets_the_same_key_after_its_window_elapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_now = [0.0]
    monkeypatch.setattr(main_module.time, "monotonic", lambda: fake_now[0])

    limiter = _FixedWindowRateLimiter(limit=1, window_seconds=60.0)
    assert limiter.check("a") == (True, 0.0)

    allowed, retry_after = limiter.check("a")
    assert allowed is False
    assert retry_after == pytest.approx(60.0)

    fake_now[0] = 61.0
    allowed, _ = limiter.check("a")
    assert allowed is True


# --------------------------------------------------------------------------- #
# 22) `_shap_lock` wait timeout (Phase 7)
# --------------------------------------------------------------------------- #


def test_shap_lock_timeout_seconds_defaults_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SHAP_LOCK_TIMEOUT_ENV, raising=False)
    assert get_shap_lock_timeout_seconds() == pytest.approx(10.0)

    monkeypatch.setenv(SHAP_LOCK_TIMEOUT_ENV, "0.25")
    assert get_shap_lock_timeout_seconds() == pytest.approx(0.25)


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1"])
def test_shap_lock_timeout_seconds_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(SHAP_LOCK_TIMEOUT_ENV, value)
    with pytest.raises(ShapLockConfigurationError):
        get_shap_lock_timeout_seconds()


def test_shap_for_transformed_raises_when_the_lock_cannot_be_acquired_in_time(
    bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The WAIT is bounded; this does not touch the computation itself (see below)."""
    monkeypatch.setenv(SHAP_LOCK_TIMEOUT_ENV, "0.05")
    frame = bundle.prepare_row({})
    transformed = bundle._transform(frame)

    bundle._shap_lock.acquire()
    try:
        with pytest.raises(ShapLockTimeoutError):
            bundle._shap_for_transformed(transformed)
    finally:
        bundle._shap_lock.release()


def test_shap_lock_timeout_returns_503_with_retry_after_over_http(
    client: TestClient, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SHAP_LOCK_TIMEOUT_ENV, "0.05")
    bundle._shap_lock.acquire()
    try:
        response = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
    finally:
        bundle._shap_lock.release()

    assert response.status_code == 503
    assert response.headers.get("retry-after") is not None
    assert response.json() == {
        "detail": "The service is busy handling other requests. Please retry shortly."
    }


def test_shap_lock_timeout_does_not_trigger_on_the_normal_unlocked_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the lock free, even a tiny timeout must not fire — the wait is near-zero."""
    monkeypatch.setenv(SHAP_LOCK_TIMEOUT_ENV, "5")
    response = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
    assert response.status_code == 200
