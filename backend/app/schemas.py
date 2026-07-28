"""The API contract — Pydantic v2 request/response models.

WHAT IT DOES, AND WHY IT DOES IT THIS WAY
-----------------------------------------
This file answers two questions: "what may the client send?" and "what do we
return?". It holds NO knowledge of how the model works — artifact loading,
imputation and SHAP are `model.py`'s job.

1) TWO LAYERS OF VALIDATION (a critical distinction)
   The Pydantic bounds are NOT THE TRAINING RANGE. The two layers ask different
   questions:

       Pydantic (here)    -> "Is this value physically/logically possible?"
                             A wide but sensible bound. The goal is to keep junk
                             data and negative/absurd input out of the model
                             entirely.
       Guardrail (Phase 2)-> "Did the model SEE this value in training?"
                             The narrow training range (metadata.training_ranges).

   If we set the Pydantic bounds equal to the training range, the guardrail could
   never fire and Phase 2 would be pointless. Example: `witnesses` is 0-3 in
   training, but an accident with 7 witnesses is physically possible — Pydantic
   accepts it and the guardrail says "I have not seen this before". That is the
   correct behaviour.

   The one exception: for some fields the physical bound genuinely coincides with
   the training range (e.g. `incident_hour_of_the_day` really is 0-23). That is
   deliberate and was decided field by field.

   A NOTE FOR PHASE 2 — ONE-WAY BLINDNESS (reviewer MINOR-7):
   For eight fields the Pydantic LOWER bound EQUALS the training minimum, so OOD
   in the "below the training range" direction can never fire for them:

       months_as_customer (0), capital-gains (0), incident_hour_of_the_day (0),
       number_of_vehicles_involved (1), bodily_injuries (0), witnesses (0),
       injury_claim (0), property_claim (0)

   The same holds at the upper bound for two fields: `capital-loss` (0) and again
   `incident_hour_of_the_day` (23).

   This is NOT a bug but a consequence of physical reality: the number of
   witnesses cannot go below 0, the number of vehicles cannot go below 1, and in
   this dataset capital loss cannot be positive. The training set already touches
   the physical floor of those fields. The bounds were DELIBERATELY left alone —
   loosening them would mean feeding physically impossible input to the model.
   The Phase 2 guardrail must not mistake "no downward warning is possible here"
   for a gap.

2) CATEGORICAL FIELDS ARE CLOSED WITH `Literal`
   The OrdinalEncoder in the pipeline is configured with `unknown_value=-1`; it
   does not fail on an unknown category, it silently encodes it as -1 and the
   model produces a meaningless but "successful"-looking prediction. Returning a
   422 at the API level is the right answer to silent nonsense.

   The category lists match `models/metadata.json -> training_ranges` exactly.
   They are written by hand (so they are readable in the OpenAPI schema and in
   /docs), but `tests/test_api.py::test_literal_choices_match_metadata` verifies
   on every run that the two stay in sync.

3) EVERY FIELD IS OPTIONAL
   CLAUDE.md: "fields that are not supplied are filled with the median/mode from
   defaults." All 34 fields default to `None`. The filling itself does not happen
   here but in one place inside `model.py`.

4) "?" MEANS SOMETHING DIFFERENT (field absent != field is "?")
   For `collision_type`, `property_damage` and `police_report_available`, the
   source dataset encodes missingness as the string `"?"`. The `Literal` for
   those three fields accepts `"?"` as well:

       field never sent (None) -> the mode value from metadata.defaults
       field sent as "?"       -> NaN -> the pipeline's SimpleImputer

   In this model both end up at the same result, but they are not the same THING:
   the first means "no information was collected, use the default", the second
   means "the source system marks this field as unknown". Preserving the
   distinction keeps behaviour correct if the defaults or the imputer strategy
   change.

5) `extra="forbid"`
   Unknown fields are not swallowed silently. A client that writes `capitalgains`
   gets a 422; otherwise it would believe it had sent the field while actually
   being scored against the default.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# The example request from CLAUDE.md — pre-filled for "Try it out" in Swagger UI.
# --------------------------------------------------------------------------- #
PREDICT_REQUEST_EXAMPLE = {
    "incident_severity": "Major Damage",
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Side Collision",
    "auto_year": 2010,
    "number_of_vehicles_involved": 1,
    "witnesses": 2,
    "police_report_available": "YES",
    "capital-gains": 30000,
    "capital-loss": 0,
    "total_claim_amount": 55000,
}


class PredictRequest(BaseModel):
    """`POST /predict` body — the pipeline's 34 input fields, all optional.

    The field order matches `metadata.feature_list.pipeline_input_order` so the
    form in /docs reads in the same order as the model's real input schema.

    IMPORTANT (K2): `incident_date` / `policy_bind_date` do NOT exist on the API
    surface. The pipeline does not parse dates; the real input schema already
    contains `incident_year` and `policy_bind_year`. Accepting raw dates and
    deriving the year would introduce a concept the model never saw (date
    parsing, time zones, format errors) into the API. We ask for the year
    directly.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [PREDICT_REQUEST_EXAMPLE]},
    )

    @field_validator("*", mode="before")
    @classmethod
    def _reject_booleans(cls, value: Any) -> Any:
        """Stops `true` / `false` from silently becoming 1 / 0 on numeric fields
        (Codex F2-1).

        MEASURED BEHAVIOUR (before the fix):

            {"witnesses": true}          -> 200, no warning      ("1 witness")
            {"total_claim_amount": true} -> 200, OOD warning!     (1 < 1920)
            {"capital-gains": false}     -> 200, no warning      ("0 gains")

        Pydantic's default (lax) mode converted the JSON value `true` to 1 on
        numeric fields. That caused two separate problems:

          * The bool guard in `guardrails.py` was INEFFECTIVE on the HTTP path —
            the guardrail already received the value as an `int` and could not see
            that it had been a `bool`.
          * More fundamentally: `witnesses: true` is NOT meaningful input.
            Silently counting it as "1 witness" would repeat exactly the mistake
            the `Literal` rationale in this file argues against — "returning a 422
            beats silent nonsense".

        `strict=True` WAS NOT USED: that mode would also reject int -> float
        conversion and would break a perfectly valid JSON input such as
        `total_claim_amount: 55000`. The ban is kept focused on `bool` alone.
        """
        if isinstance(value, bool):
            raise ValueError(
                "boolean values are not accepted; this field expects a number or a category"
            )
        return value

    # --- Policy / customer ------------------------------------------------- #

    # Training: 0-479 months. 1200 months (=100 years) was chosen as the physical
    # ceiling; a longer customer relationship is a data entry error.
    months_as_customer: Annotated[int, Field(ge=0, le=1200)] | None = None

    # Training: 20-64. The bounds are 16 (the minimum driving age in many states)
    # to 120 (the ceiling of a human lifespan). A 70-year-old policyholder is
    # valid input; that the model has not seen one is a separate thing for the
    # guardrail to say.
    age: Annotated[int, Field(ge=16, le=120)] | None = None

    policy_state: Literal["IL", "IN", "OH"] | None = None

    policy_csl: Literal["100/300", "250/500", "500/1000"] | None = None

    # Training: 500-2000. A deductible cannot be negative; 100k is a reasonable
    # upper ceiling.
    policy_deductable: Annotated[float, Field(ge=0, le=100_000)] | None = None

    # Training: 433.33-2047.59. A premium cannot be negative. Fractional values
    # are accepted (the source column is already float64).
    policy_annual_premium: Annotated[float, Field(ge=0, le=100_000)] | None = None

    # The training range runs from -1,000,000 to 10,000,000. NOTE: the training
    # data contains NEGATIVE umbrella_limit values (a data quality problem). Had
    # we set the lower bound to 0, we would reject with a 422 the very values the
    # model saw in training — a Pydantic bound must be a SUPERSET of the training
    # range.
    umbrella_limit: Annotated[float, Field(ge=-10_000_000, le=100_000_000)] | None = None

    insured_sex: Literal["FEMALE", "MALE"] | None = None

    insured_education_level: (
        Literal["Associate", "College", "High School", "JD", "MD", "Masters", "PhD"] | None
    ) = None

    insured_occupation: (
        Literal[
            "adm-clerical",
            "armed-forces",
            "craft-repair",
            "exec-managerial",
            "farming-fishing",
            "handlers-cleaners",
            "machine-op-inspct",
            "other-service",
            "priv-house-serv",
            "prof-specialty",
            "protective-serv",
            "sales",
            "tech-support",
            "transport-moving",
        ]
        | None
    ) = None

    insured_hobbies: (
        Literal[
            "base-jumping",
            "basketball",
            "board-games",
            "bungie-jumping",
            "camping",
            "chess",
            "cross-fit",
            "dancing",
            "exercise",
            "golf",
            "hiking",
            "kayaking",
            "movies",
            "paintball",
            "polo",
            "reading",
            "skydiving",
            "sleeping",
            "video-games",
            "yachting",
        ]
        | None
    ) = None

    insured_relationship: (
        Literal["husband", "not-in-family", "other-relative", "own-child", "unmarried", "wife"]
        | None
    ) = None

    # Training: 0-100,500. Capital GAINS cannot be negative by definition (losses
    # are held in a separate column).
    # `alias`: the JSON key is hyphenated (`capital-gains`) because that is the
    # pipeline's column name; a hyphen is not a valid Python identifier.
    capital_gains: Annotated[float, Field(ge=0, le=10_000_000)] | None = Field(
        default=None, alias="capital-gains"
    )

    # Training: -111,100 to 0. In this dataset a loss is encoded with a NEGATIVE
    # sign; a positive value would be a sign error and would send the model a
    # signal in the wrong direction. `le=0` is therefore a deliberate logical
    # constraint.
    capital_loss: Annotated[float, Field(ge=-10_000_000, le=0)] | None = Field(
        default=None, alias="capital-loss"
    )

    # --- Incident ---------------------------------------------------------- #

    incident_type: (
        Literal["Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"]
        | None
    ) = None

    # "?" = unknown in the source system -> NaN -> the pipeline's imputer (see 4).
    collision_type: (
        Literal["Front Collision", "Rear Collision", "Side Collision", "?"] | None
    ) = None

    incident_severity: (
        Literal["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"] | None
    ) = None

    authorities_contacted: Literal["Ambulance", "Fire", "Other", "Police"] | None = None

    incident_state: Literal["NC", "NY", "OH", "PA", "SC", "VA", "WV"] | None = None

    incident_city: (
        Literal[
            "Arlington",
            "Columbus",
            "Hillsdale",
            "Northbend",
            "Northbrook",
            "Riverwood",
            "Springfield",
        ]
        | None
    ) = None

    # EXCEPTION: here the physical bound is identical to the training range
    # (0-23), because an hour of the day is 0-23 by definition. A numeric OOD
    # warning can never fire on this field; that is not a gap but the nature of
    # the field.
    incident_hour_of_the_day: Annotated[int, Field(ge=0, le=23)] | None = None

    # Training: 1-4. A pile-up can involve many more; the ceiling is 100.
    number_of_vehicles_involved: Annotated[int, Field(ge=1, le=100)] | None = None

    property_damage: Literal["NO", "YES", "?"] | None = None

    # Training: 0-2. An accident with many injuries is possible; ceiling 20.
    bodily_injuries: Annotated[int, Field(ge=0, le=20)] | None = None

    # Training: 0-3. Up to 10 witnesses is plausible; the guardrail warns above 3.
    witnesses: Annotated[int, Field(ge=0, le=10)] | None = None

    police_report_available: Literal["NO", "YES", "?"] | None = None

    # --- Claim amounts ----------------------------------------------------- #
    # None of these can be negative (they are amounts paid). The ceiling of
    # 10,000,000 is generous: the largest claim in training is ~115k, but large
    # commercial losses can reach that order. Amounts are accepted as floats
    # (cents are realistic).

    total_claim_amount: Annotated[float, Field(ge=0, le=10_000_000)] | None = None
    injury_claim: Annotated[float, Field(ge=0, le=10_000_000)] | None = None
    property_claim: Annotated[float, Field(ge=0, le=10_000_000)] | None = None
    vehicle_claim: Annotated[float, Field(ge=0, le=10_000_000)] | None = None

    # --- Vehicle / date-derived fields ------------------------------------- #

    auto_make: (
        Literal[
            "Accura",
            "Audi",
            "BMW",
            "Chevrolet",
            "Dodge",
            "Ford",
            "Honda",
            "Jeep",
            "Mercedes",
            "Nissan",
            "Saab",
            "Suburu",
            "Toyota",
            "Volkswagen",
        ]
        | None
    ) = None

    # All year fields use 1900-2100: a sensible calendar window in which a car or
    # a policy could exist. The training ranges are far narrower (auto_year
    # 1995-2015, policy_bind_year 1990-2015, incident_year fixed at 2015) — the
    # gap is deliberate and is the guardrail's job.
    auto_year: Annotated[int, Field(ge=1900, le=2100)] | None = None
    incident_year: Annotated[int, Field(ge=1900, le=2100)] | None = None
    policy_bind_year: Annotated[int, Field(ge=1900, le=2100)] | None = None


# --------------------------------------------------------------------------- #
# /predict response
# --------------------------------------------------------------------------- #


class ShapValue(BaseModel):
    """One feature's SHAP contribution — exactly as in the CLAUDE.md contract.

    `value` and `base_value` live in LOG-ODDS (raw margin) space, not probability.
    They sum to the model's raw margin: sum(value) + base_value = raw margin, and
    sigmoid(raw margin) = fraud_probability. That additivity is what lets the
    frontend's waterfall chart be mathematically correct.
    """

    feature: str = Field(description="Clean feature name (no cat__/remainder__ prefix).")
    value: float = Field(description="This feature's log-odds contribution.")
    base_value: float = Field(description="The model's base log-odds value (identical in every item).")


class PredictResponse(BaseModel):
    """`POST /predict` response — the 4 fields of the CLAUDE.md contract, no more.

    DELIBERATELY ABSENT: any echo of the input, the model version, internal IDs.
    The narrower the response, the smaller the leak surface; no field outside the
    contract is ever added.
    """

    fraud_probability: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    shap_values: list[ShapValue] = Field(
        description="All 34 features, sorted by descending abs(value). The frontend decides how many to show."
    )
    # A CRITICAL NOTE FOR PHASE 2 — THE GUARDRAIL CAN ONLY PRODUCE NUMERIC OOD
    # (Codex C-4). Categorical fields are closed with `Literal` to the values seen
    # in training: an unknown category gets a 422 and NEVER REACHES the guardrail.
    # So the scenario "I sent a `policy_state` outside the training set and expect
    # a warning" is impossible — a 422 comes back instead.
    #
    # This is deliberate and correct: `OrdinalEncoder` is configured with
    # `unknown_value=-1` and would silently encode an unknown category as -1,
    # producing a meaningless but "successful"-looking prediction (see point 2 at
    # the top of this file). But the field name and description must not hide that
    # boundary: if the Phase 4 UI banner is written on the assumption that
    # categorical OOD warnings arrive, it will be silently wrong.
    out_of_distribution_warnings: list[str] = Field(
        description=(
            "Names of fields whose values fall outside the range the model saw in "
            "training. Only numeric fields ACTUALLY SENT IN THE REQUEST are checked: "
            "fields that were not supplied are filled with the training median/mode "
            "and are therefore inside the range by definition. Categorical fields "
            "never appear here — an invalid category produces a 422, not a warning. "
            "The ordering is meaningful: fields the model actually uses come first, "
            "and the ones it never uses (feature_influence.has_influence=false) come "
            "last. A warning does not block the prediction; the score is still "
            "returned and should be read with caution."
        )
    )


# --------------------------------------------------------------------------- #
# /health response
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    """Simple healthcheck.

    `protected_namespaces=()`: by default Pydantic v2 warns about field names
    starting with `model_` because they may clash with its own API. Our field
    names (`model_loaded`, `model_version`) come from the metadata contract, so
    rather than muting the warning we turn the namespace protection off.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    # Not just "the process is up" but "the artifact really loaded".
    model_loaded: bool
    # Answers "which artifact is running?" in a single request after a deploy.
    model_version: str


# --------------------------------------------------------------------------- #
# /model-info response — AN ALLOWLIST PROJECTION
# --------------------------------------------------------------------------- #
#
# metadata.json IS NOT RETURNED AS-IS. The models below establish an explicit
# allowlist: each sub-model declares only the permitted fields, and thanks to
# `extra="ignore"` everything else in the metadata is dropped silently.
#
# Why an allowlist rather than a blocklist: if train_pipeline.py adds a new field
# to the metadata tomorrow (an internal cost metric, say), a blocklist would make
# that field public automatically. With an allowlist it stays in until someone
# adds it explicitly.
#
# What is left out, and why:
#   preprocessing_contract -> an implementation detail; useless to a client and
#                             it only exposes the internal architecture.
#   model_params           -> hyperparameters (n_estimators, learning_rate, ...).
#   dataset.source_file    -> a server-side filename; unnecessary.


class MetricsInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_pr_auc: float
    test_pr_auc: float
    metric_name: str


# CAUTION — THE NOTES BELOW ARE DELIBERATELY COMMENTS, NOT DOCSTRINGS:
# Pydantic model docstrings become the `description` in the OpenAPI schema, and
# `/openapi.json` is a public surface. Naming the metadata keys we left OUT of the
# allowlist inside a docstring would carry the internal structure we carefully
# removed from `/model-info` back into the public schema through the back door.
# The educational value for someone reading the source is the same; for OpenAPI
# it is invisible.
#
#   DatasetInfo      -> the metadata's source filename field is DELIBERATELY
#                       absent; a server-side filename is of no use to a client.
#   FeatureListInfo  -> the transformed column-name lists used for the SHAP
#                       mapping are NOT in the allowlist: they are an internal
#                       detail and already appear with clean names in the
#                       `/predict` response.
class DatasetInfo(BaseModel):
    """Size and split information for the training dataset."""

    model_config = ConfigDict(extra="ignore")

    n_rows: int
    n_train: int
    n_test: int
    test_size: float
    random_state: int
    positive_rate_train: float
    positive_rate_test: float


class FeatureListInfo(BaseModel):
    """The feature section of the model card."""

    model_config = ConfigDict(extra="ignore")

    pipeline_input_order: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    # NOTE: this lists PII column NAMES (policy_number, insured_zip,
    # incident_location). That is not a leak but the opposite: a statement that
    # "these fields never reached the model". A leak would be returning the PII
    # *values* — which is impossible anyway, since the model never saw those
    # columns.
    dropped_columns: list[str]
    target: str


class TrainingRangeInfo(BaseModel):
    """The "what did we see in training" record the Phase 2 guardrail relies on.

    Numeric fields carry min/max, categorical fields carry categories; the other
    stays None. The `int | float` union keeps integer bounds from appearing as
    2015.0 in the response.
    """

    model_config = ConfigDict(extra="ignore")

    type: Literal["numeric", "categorical"]
    dtype: str
    min: int | float | None = None
    max: int | float | None = None
    categories: list[str] | None = None


class DefaultInfo(BaseModel):
    """The median/mode value used for fields that were not supplied.

    Public so the frontend can show it as a form placeholder: the user should be
    able to see the answer to "what was the field I left blank filled in with?".
    """

    model_config = ConfigDict(extra="ignore")

    value: int | float | str
    dtype: str
    strategy: Literal["median", "mode"]


class FeatureInfluenceEntry(BaseModel):
    """One feature's MEASURED influence in the trained model."""

    model_config = ConfigDict(extra="ignore")

    split_count: int = Field(description="Number of splits made on this column across the trees.")
    gain: float = Field(description="Total gain of those splits.")
    has_influence: bool = Field(description="split_count > 0 — did the model use this feature?")


class FeatureInfluenceSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_features: int
    n_with_influence: int
    n_without_influence: int
    features_without_influence: list[str]


class FeatureInfluenceInfo(BaseModel):
    """A measurement of which features actually affect a prediction.

    It feeds the "this field does not affect the model" badge on the Phase 3 form.
    The frontend is FORBIDDEN from embedding this list: retraining the model
    changes it, and an embedded copy would start lying silently. The artifact is
    the single source of truth, and this endpoint the single distribution channel.

    The keys match the API field names exactly (`capital-gains` hyphenated), so
    the frontend needs no additional mapping table.
    """

    model_config = ConfigDict(extra="ignore")

    description: str
    measured_from: str
    interpretation_note: str
    summary: FeatureInfluenceSummary
    features: dict[str, FeatureInfluenceEntry]


class FairnessAttributeInfo(BaseModel):
    """A protected/proxy attribute — DECLARATION and MEASUREMENT side by side.

    `used_as_model_feature` is a declaration: was the feature given to the model?
    `has_influence` is a measurement: did the trained trees actually use it?
    They are different things, and a model card has to show both — showing only
    the declaration misleads the reader.
    """

    model_config = ConfigDict(extra="ignore")

    feature: str
    basis: str
    rationale: str
    # DECLARATION: the feature was given to the model as an input.
    used_as_model_feature: bool
    # MEASUREMENT: computed from the booster, never hand-written.
    split_count: int
    has_influence: bool
    # Present only on `protected_attributes_used_as_features` entries.
    severity: str | None = None


class FairnessInfo(BaseModel):
    """The fairness section of the model card — Phase 5 renders this as a table.

    KEPT in the allowlist (K9). Hiding that the model uses protected attributes as
    features would be wrong even for a demo.
    """

    model_config = ConfigDict(extra="ignore")

    status: str
    # The text explaining the difference between "declaration" and "measurement"
    # to the reader; the model card page prints it verbatim.
    field_semantics: str
    protected_attributes_used_as_features: list[FairnessAttributeInfo]
    proxy_risk_attributes: list[FairnessAttributeInfo]
    audit_performed: bool
    audit_metrics_computed: list[str]
    intended_use: str
    production_requirements: list[str]
    notes: str


class RiskThresholdsInfo(BaseModel):
    """An explicit statement of how the `risk_level` label is produced.

    The frontend reads the thresholds from here rather than copying them; the
    single source of truth is the constants in `model.py`.
    """

    model_config = ConfigDict(extra="ignore")

    low_below: float
    high_at_or_above: float
    rule: str


class CalibrationInfo(BaseModel):
    """An explicit warning that the probabilities are NOT calibrated.

    Saying this on the model card is mandatory: the model was trained with an
    imbalanced class weight, so `fraud_probability` does NOT mean "71 out of 100
    such cases are fraudulent". It is reliable for ranking, not as an absolute
    probability. Without this warning an insurance client will misread the number.
    """

    model_config = ConfigDict(extra="ignore")

    calibrated: bool
    method: str
    warning: str


class ModelInfoResponse(BaseModel):
    """`GET /model-info` response — the data source for the Phase 5 model card page."""

    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    model_name: str
    model_version: str
    algorithm: str
    trained_at: str
    metrics: MetricsInfo
    dataset: DatasetInfo
    feature_list: FeatureListInfo
    feature_influence: FeatureInfluenceInfo
    training_ranges: dict[str, TrainingRangeInfo]
    defaults: dict[str, DefaultInfo]
    fairness: FairnessInfo
    library_versions: dict[str, str]
    risk_thresholds: RiskThresholdsInfo
    probability_calibration: CalibrationInfo
