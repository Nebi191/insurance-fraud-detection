"""Phase 2 — guardrail (OOD detection) tests.

TEST PHILOSOPHY
---------------
The danger with a guardrail is that it looks like it is "working" while doing
nothing at all: a function that returns an empty list on every request stays
happily green. So the tests here pin down both directions — does a warning fire
where it should, and does it stay silent where it should not?

The guardrail's thresholds must also NOT BE HARD-CODED. If they were, the
warnings would start lying silently once the model was retrained; we test that
with a metadata mutation.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.guardrails import Guardrail
from app.model import ModelBundle
from app.schemas import PREDICT_REQUEST_EXAMPLE, PredictRequest


@pytest.fixture(scope="session")
def guardrail(bundle: ModelBundle) -> Guardrail:
    """The guardrail object the application actually uses."""
    return bundle.guardrail


# --------------------------------------------------------------------------- #
# 1) Basic behaviour
# --------------------------------------------------------------------------- #


def test_values_inside_the_training_range_produce_no_warning(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """No value INSIDE the training range may produce a warning.

    The MIDPOINT of the range is tried for every numeric field: proof that the
    guardrail has not degenerated into something that warns about everything.
    """
    ranges = metadata["training_ranges"]
    payload = {
        name: (spec["min"] + spec["max"]) / 2
        for name, spec in ranges.items()
        if spec["type"] == "numeric"
    }
    assert len(payload) == 18, "the number of numeric fields has changed"
    assert guardrail.check(payload) == []


def test_every_numeric_field_can_be_flagged(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """OOD can be triggered for EACH of the 18 numeric fields.

    If a single field's check were forgotten, or read under the wrong key (e.g.
    `capital_gains` vs `capital-gains`), this catches it.
    """
    ranges = metadata["training_ranges"]
    numeric = {n: s for n, s in ranges.items() if s["type"] == "numeric"}

    for name, spec in numeric.items():
        above = guardrail.check({name: spec["max"] + 1})
        below = guardrail.check({name: spec["min"] - 1})
        assert above == [name], f"{name}: above the upper bound was not flagged ({above})"
        assert below == [name], f"{name}: below the lower bound was not flagged ({below})"


def test_range_boundaries_are_inclusive(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Extreme values SEEN in training must not produce a warning.

    `min` and `max` are values actually observed in the training data; calling
    them "out of distribution" would mean flagging data the model saw as data it
    did not see.
    """
    ranges = metadata["training_ranges"]
    for name, spec in ranges.items():
        if spec["type"] != "numeric":
            continue
        assert guardrail.check({name: spec["min"]}) == [], f"{name}: min produced a warning"
        assert guardrail.check({name: spec["max"]}) == [], f"{name}: max produced a warning"


def test_constant_field_flags_anything_but_the_single_observed_value(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """`incident_year` is CONSTANT in training (min=max=2015) — anything else is OOD.

    Edge case: when min equals max, the range collapses to a single point. That is
    the reality of this dataset (every incident is in 2015) and proof that the
    guardrail works correctly even on the narrowest possible range.
    """
    spec = metadata["training_ranges"]["incident_year"]
    assert spec["min"] == spec["max"] == 2015

    assert guardrail.check({"incident_year": 2015}) == []
    assert guardrail.check({"incident_year": 2014}) == ["incident_year"]
    assert guardrail.check({"incident_year": 2016}) == ["incident_year"]


# --------------------------------------------------------------------------- #
# 2) The difference between "not sent" and "out of range"
# --------------------------------------------------------------------------- #


def test_missing_fields_are_never_flagged(guardrail: Guardrail) -> None:
    """A field that was not sent produces no warning — it gets a default.

    Without this distinction an empty request (`{}`) would print a warning for all
    34 fields and the field would become worthless. The `metadata.defaults` values
    are inside the training range by definition; marking a field the user never
    touched as problematic would be wrong.
    """
    assert guardrail.check({}) == []


def test_explicit_null_is_treated_as_missing_not_as_zero(guardrail: Guardrail) -> None:
    """A field explicitly sent as `null` also counts as "not sent".

    Feeding `None` into a numeric comparison would raise a TypeError; silently
    treating it as 0 would produce a FALSE warning on a field with a negative
    range such as `capital-loss`. `prepare_row()` makes the same distinction
    (model.py K3).
    """
    assert guardrail.check({"age": None, "witnesses": None}) == []


def test_defaults_never_trigger_a_warning(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Every default value lies inside the training range.

    This is a consistency check: `defaults` is derived from the median/mode, so it
    MUST be inside the range. If it were not, the guardrail and the defaults would
    contradict each other and even an empty request would produce warnings.
    """
    defaults = {name: spec["value"] for name, spec in metadata["defaults"].items()}
    assert guardrail.check(defaults) == []


# --------------------------------------------------------------------------- #
# 3) Scope boundary: categorical fields (Codex C-4)
# --------------------------------------------------------------------------- #


def test_categorical_fields_are_out_of_scope(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Categorical fields are NOT in the guardrail's scope — deliberately.

    The `Literal` in `schemas.py` cuts an unknown category off with a 422, so a
    categorical OOD value can NEVER REACH here. Writing a check would produce dead
    code. This test puts the boundary explicitly on record: if the scope narrows
    or widens, this speaks up.
    """
    categorical = [
        name for name, spec in metadata["training_ranges"].items()
        if spec["type"] == "categorical"
    ]
    assert len(categorical) == 16

    for name in categorical:
        assert name not in guardrail.numeric_bounds
        # Even given junk, the guardrail stays silent on a categorical field.
        assert guardrail.check({name: "VALUE-NOT-SEEN-IN-TRAINING"}) == []


def test_unknown_category_is_rejected_by_the_api_not_by_the_guardrail(
    client: TestClient,
) -> None:
    """The real behaviour of categorical OOD: a 422, not a warning.

    If the Phase 4 UI banner is written on the assumption that categorical OOD
    warnings arrive, it will be silently wrong. This test breaks that assumption
    up front.
    """
    response = client.post("/predict", json={"policy_state": "TX"})
    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["loc"] == ["body", "policy_state"]


# --------------------------------------------------------------------------- #
# 4) Ordering: fields the model actually looks at come first
# --------------------------------------------------------------------------- #


def test_influential_fields_are_listed_before_dead_ones(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Influential fields at the front of the list, dead ones at the back.

    An out-of-range value for `umbrella_limit` (split_count=0) GENUINELY does not
    affect the score; for `age` (live) it does. Listing them in mixed order would
    convince the reader they carry equal weight.
    """
    influence = metadata["feature_influence"]["features"]
    order = metadata["feature_list"]["pipeline_input_order"]

    # THE CHOICE OF FIELDS MATTERS: the dead fields must come BEFORE the live ones
    # in the input order, otherwise the test passes even without the sorting (it
    # becomes tautological — the first version fell into exactly that trap and was
    # caught by mutation testing).
    #
    #   policy_deductable  position  4  DEAD      witnesses   position 24  LIVE
    #   umbrella_limit     position  6  DEAD      auto_year   position 31  LIVE
    #
    # Without the sorting the result would be ["policy_deductable",
    # "umbrella_limit", "witnesses", "auto_year"].
    for dead in ("policy_deductable", "umbrella_limit"):
        assert influence[dead]["has_influence"] is False
    for alive in ("witnesses", "auto_year"):
        assert influence[alive]["has_influence"] is True
    assert order.index("policy_deductable") < order.index("witnesses")
    assert order.index("umbrella_limit") < order.index("auto_year")

    warnings = guardrail.check(
        {
            "policy_deductable": 90_000,
            "umbrella_limit": 99_000_000,
            "witnesses": 9,
            "auto_year": 2050,
        }
    )
    assert warnings == ["witnesses", "auto_year", "policy_deductable", "umbrella_limit"], (
        f"live fields were not moved to the front: {warnings}"
    )


def test_ordering_is_deterministic(guardrail: Guardrail) -> None:
    """Same input -> same order. It must not depend on dict insertion order."""
    first = guardrail.check({"age": 110, "witnesses": 9, "auto_year": 2050})
    second = guardrail.check({"auto_year": 2050, "witnesses": 9, "age": 110})
    assert first == second
    assert len(first) == 3


# --------------------------------------------------------------------------- #
# 5) Thresholds come from the metadata, they are NOT hard-coded
# --------------------------------------------------------------------------- #


def test_thresholds_come_from_metadata_not_from_hardcoded_values(
    bundle: ModelBundle,
) -> None:
    """If the range in the metadata changes, the guardrail's behaviour must change too.

    Were the thresholds hard-coded, the warnings would start lying silently after a
    retrain — saying "I have not seen this value" when in fact it had. The test
    narrows the range and shows that a value which previously passed cleanly is
    now flagged.
    """
    # In the real range, age 30 is entirely normal (20-64).
    assert bundle.guardrail.check({"age": 30}) == []

    narrowed = copy.deepcopy(bundle.metadata)
    narrowed["training_ranges"]["age"]["min"] = 40
    narrowed["training_ranges"]["age"]["max"] = 50

    probe = Guardrail(narrowed)
    assert probe.check({"age": 30}) == ["age"]
    assert probe.check({"age": 45}) == []

    # The original guardrail must be unaffected (no shared-state leak).
    assert bundle.guardrail.check({"age": 30}) == []


# --------------------------------------------------------------------------- #
# 6) Silent-pass traps
# --------------------------------------------------------------------------- #


def test_nan_is_flagged_instead_of_silently_passing(guardrail: Guardrail) -> None:
    """NaN must not count as "inside the range" and pass silently.

    `nan < min` and `nan > max` are BOTH False. Without a dedicated check, NaN
    would look like a clean request — the guardrail's most insidious silent
    failure.
    """
    assert guardrail.check({"age": math.nan}) == ["age"]


def test_infinity_is_flagged(guardrail: Guardrail) -> None:
    """An infinite value exceeds every upper bound."""
    assert guardrail.check({"age": math.inf}) == ["age"]
    assert guardrail.check({"capital-loss": -math.inf}) == ["capital-loss"]


def test_huge_integers_are_flagged_not_crashed(guardrail: Guardrail) -> None:
    """A huge `int` that does not fit in a float must warn, not raise (Codex F2-3).

    `float(10**10000)` raises `OverflowError`. It is unreachable through HTTP
    (Pydantic's bounds are far tighter and the JSON encoder blows up first), but
    this module can also be called without HTTP, and an OOD detection layer
    crashing on oversized input would be ironic.
    """
    assert guardrail.check({"age": 10**10000}) == ["age"]
    assert guardrail.check({"capital-loss": -(10**10000)}) == ["capital-loss"]


def test_booleans_are_rejected_over_http_not_silently_coerced(
    client: TestClient,
) -> None:
    """`true` must NOT be SCORED as 1 on a numeric field (Codex F2-1).

    MEASURED BEHAVIOUR (before the fix): Pydantic's lax mode converted `true` to
    1, so the bool guard in `guardrails.py` never engaged on the HTTP path — the
    guardrail already received the value as an `int`.

        {"witnesses": true}          -> 200, no warning
        {"total_claim_amount": true} -> 200, OOD warning (1 < 1920)
        {"capital-gains": false}     -> 200, no warning

    Writing this test at the ENDPOINT level rather than the guardrail level is
    essential: called directly, `Guardrail.check()` already skipped the bool, and
    that test stayed green while hiding the real hole.
    """
    for field in ("witnesses", "total_claim_amount", "capital-gains", "bodily_injuries"):
        for value in (True, False):
            response = client.post("/predict", json={field: value})
            assert response.status_code == 422, (
                f"{field}={value} was accepted ({response.status_code}) — "
                "a boolean is being silently coerced into a number"
            )

    # Real numbers must be unaffected.
    assert client.post("/predict", json={"witnesses": 1}).status_code == 200
    assert client.post("/predict", json={"capital-gains": 0}).status_code == 200


def test_booleans_are_not_treated_as_numbers(guardrail: Guardrail) -> None:
    """`True` must not be silently compared as 1.

    In Python `bool` is a subclass of `int`: `isinstance(True, int)` is true.
    Without the check, a client sending `witnesses: true` would be treated as
    reporting "1 witness". This layer does not decide — Pydantic already returns a
    422 — but the guardrail must not make a wrong comparison when called directly
    either.
    """
    assert guardrail.check({"witnesses": True}) == []
    assert guardrail.check({"age": False}) == []


def test_non_numeric_values_are_skipped_not_crashed(guardrail: Guardrail) -> None:
    """A string on a numeric field must not crash the guardrail (that is Pydantic's job)."""
    assert guardrail.check({"age": "thirty"}) == []
    assert guardrail.check({"witnesses": [1, 2]}) == []


def test_unknown_keys_in_payload_are_ignored(guardrail: Guardrail) -> None:
    """An unknown key must not bring the guardrail down.

    `extra="forbid"` on the API already cuts this off; because the guardrail can
    also be called without HTTP, it has to be robust on its own.
    """
    assert guardrail.check({"no_such_field_exists": 999, "age": 110}) == ["age"]


# --------------------------------------------------------------------------- #
# 7) End to end: together with the HTTP layer
# --------------------------------------------------------------------------- #


def test_warning_does_not_block_the_prediction(client: TestClient) -> None:
    """An OOD warning does NOT block the prediction — the score still comes back.

    The guardrail's design decision: flag, do not reject. For an adjuster, "a
    rejected claim" and "a claim the model is unsure about" are not the same thing.
    """
    response = client.post("/predict", json={"witnesses": 9, "auto_year": 2050})
    assert response.status_code == 200
    body = response.json()

    assert set(body["out_of_distribution_warnings"]) == {"witnesses", "auto_year"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert len(body["shap_values"]) == 34


def test_training_extremes_pass_through_both_layers(
    client: TestClient, metadata: dict[str, Any]
) -> None:
    """The training extremes must pass cleanly through both Pydantic and the guardrail.

    Proof that the two layers do not collide: a Pydantic bound has to be a SUPERSET
    of the training range (point 1 at the top of schemas.py). If a field's Pydantic
    bound fell inside the training range, we would reject with a 422 a value the
    model has actually seen.
    """
    ranges = metadata["training_ranges"]
    numeric = {n: s for n, s in ranges.items() if s["type"] == "numeric"}

    for bound in ("min", "max"):
        payload = {name: spec[bound] for name, spec in numeric.items()}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"the {bound} extremes were rejected: {response.text}"
        assert response.json()["out_of_distribution_warnings"] == []


def test_shap_values_are_still_correct_when_a_warning_fires(
    client: TestClient, bundle: ModelBundle
) -> None:
    """Producing a warning must not disturb the prediction path.

    The guardrail was added inside `predict()`; SHAP additivity must still hold
    (i.e. the guardrail does not interfere with the computation).
    """
    payload = {"witnesses": 9, "age": 110, "incident_severity": "Major Damage"}
    body = client.post("/predict", json=payload).json()

    assert body["out_of_distribution_warnings"]
    margin = sum(item["value"] for item in body["shap_values"])
    margin += body["shap_values"][0]["base_value"]
    reconstructed = 1.0 / (1.0 + math.exp(-margin))
    assert reconstructed == pytest.approx(body["fraud_probability"], abs=1e-9)

    # The model layer called directly must produce the same warnings.
    direct = bundle.predict(PredictRequest(**payload).model_dump(by_alias=True))
    assert direct["out_of_distribution_warnings"] == body["out_of_distribution_warnings"]


def test_example_request_from_claude_md_is_clean(client: TestClient) -> None:
    """The example request in CLAUDE.md is entirely inside the training range."""
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert body["out_of_distribution_warnings"] == []


# --------------------------------------------------------------------------- #
# 8) ONE-WAY BLINDNESS — the guardrail's real reach over HTTP
# --------------------------------------------------------------------------- #


# Fields whose Pydantic LOWER bound equals the training minimum: for these,
# "below the training range" OOD can NEVER be triggered through the API, because a
# smaller value already gets a 422.
#
# This is NOT a bug but a consequence of physical reality: the number of witnesses
# cannot go below 0, the number of vehicles cannot go below 1. The training set
# already touches the physical floor of these fields. Loosening the bounds would
# mean feeding physically impossible input to the model (see the "ONE-WAY
# BLINDNESS" note at the top of `schemas.py`).
BLIND_BELOW = {
    "months_as_customer",
    "capital-gains",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "injury_claim",
    "property_claim",
}

# The same situation at the upper bound.
BLIND_ABOVE = {"capital-loss", "incident_hour_of_the_day"}


def _pydantic_bounds(column: str) -> tuple[float, float]:
    """The field's PUBLISHED JSON Schema bounds (the contract the frontend sees)."""
    properties = PredictRequest.model_json_schema()["properties"]
    branches = [
        branch
        for branch in properties[column].get("anyOf", [properties[column]])
        if branch.get("type") != "null"
    ]
    assert len(branches) == 1, f"unexpected schema branch for '{column}'"
    return float(branches[0]["minimum"]), float(branches[0]["maximum"])


def test_one_way_blindness_matches_the_documented_list(
    metadata: dict[str, Any],
) -> None:
    """Which direction the guardrail cannot fire in is MEASURED, not assumed.

    `schemas.py` declares this list in a comment. If the declaration and reality
    diverge (e.g. a field's Pydantic bound changes), this speaks up — otherwise
    "why does this field never warn?" would be a much more expensive question to
    answer in Phase 4.
    """
    ranges = metadata["training_ranges"]
    measured_below: set[str] = set()
    measured_above: set[str] = set()

    for column, spec in ranges.items():
        if spec["type"] != "numeric":
            continue
        low, high = _pydantic_bounds(column)
        if low == float(spec["min"]):
            measured_below.add(column)
        if high == float(spec["max"]):
            measured_above.add(column)

    assert measured_below == BLIND_BELOW, (
        "the list of fields blind below has changed — the note in schemas.py needs updating"
    )
    assert measured_above == BLIND_ABOVE, (
        "the list of fields blind above has changed — the note in schemas.py needs updating"
    )


def test_blind_direction_returns_422_instead_of_a_warning(client: TestClient) -> None:
    """In a blind direction a 422 arrives, NOT a warning — end-to-end proof of the behaviour.

    It prevents the impression that the guardrail is "silently not working": the
    reason no warning is produced in that direction is not that the guardrail
    failed, but that the request never reached it.
    """
    # witnesses is 0-3 in training; below 0 is physically impossible.
    assert client.post("/predict", json={"witnesses": -1}).status_code == 422
    # But in the UPWARD direction the guardrail is active.
    body = client.post("/predict", json={"witnesses": 9})
    assert body.status_code == 200
    assert body.json()["out_of_distribution_warnings"] == ["witnesses"]


def test_fields_blind_in_both_directions_can_never_warn(client: TestClient) -> None:
    """`incident_hour_of_the_day` is blind in both directions — it can never warn.

    The hour of the day is 0-23 by definition and the training set covers the whole
    range. So the concept of OOD is meaningless for this field. Phase 4 must not
    put a "a warning may appear" indicator next to it.
    """
    assert "incident_hour_of_the_day" in BLIND_BELOW & BLIND_ABOVE

    for hour in (0, 12, 23):
        body = client.post("/predict", json={"incident_hour_of_the_day": hour}).json()
        assert body["out_of_distribution_warnings"] == []

    for invalid in (-1, 24):
        assert client.post(
            "/predict", json={"incident_hour_of_the_day": invalid}
        ).status_code == 422
