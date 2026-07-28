"""Out-of-distribution (OOD) detection — "did the model see this value in training?".

WHAT IT DOES, AND WHY IT DOES IT THIS WAY
-----------------------------------------

1) WHY IT EXISTS
   `predict_proba` returns a number for every input. It returns one even when the
   input sits in a region the training data never reached — there is NO output
   that means "this is unfamiliar to me". A score of 0.71 comes back,
   `risk_level` says "high", the SHAP chart renders; everything looks normal, but
   there is not a single observation behind that 0.71.

   With tree models this is especially insidious. LightGBM splits on thresholds
   like `auto_year <= 2012.5`, and since no training row exceeds 2015, NO
   threshold can sit above 2015. The result: a 2016 vehicle and a 2050 vehicle
   land in the same leaf and receive the same contribution. The model cannot tell
   "very new vehicle" apart — but you would think it could.

   This module turns that silence into a signal. It never BLOCKS the prediction —
   the score is still returned, with a note saying "I have never seen these
   fields before". For an insurance adjuster that is a critical distinction: a
   rejected claim and a claim the model is unsure about are not the same thing.

2) NUMERIC FIELDS ONLY — AND THAT IS NOT A GAP
   Categorical fields are closed to the values seen in training via `Literal` in
   `schemas.py`: an invalid category gets a 422 and NEVER REACHES the guardrail.
   So a categorical OOD check would be DEAD CODE — a branch that can never be
   triggered through the API and therefore can never genuinely be tested.

   This is a deliberate design: `OrdinalEncoder` is configured with
   `unknown_value=-1`, so it would silently encode an unknown category as -1 and
   produce a meaningless but "successful"-looking prediction. Returning a 422 is
   safer. But the boundary itself must not be hidden: the field description on
   the `/predict` response and the model card both state it plainly.

3) ONLY FIELDS THAT WERE ACTUALLY SENT ARE CHECKED
   Fields that were not supplied get filled from `metadata.defaults` (median or
   mode), and those values are BY DEFINITION inside the training range. Emitting
   a warning for a filled field would mark something the user never touched as
   problematic — and every empty request would print 34 warnings.

   The distinction is made on `None`: the field was never sent, or was explicitly
   sent as `null` -> no check. That is exactly the same distinction
   `prepare_row()` uses for default filling (see `model.py` K3/K4).

4) THE ORDERING IS MEANINGFUL: FIELDS THE MODEL ACTUALLY LOOKS AT COME FIRST
   16 of this model's 34 features have a split count of zero — the model never
   looks at them and their SHAP contributions are exactly 0.0. When an
   out-of-range value arrives for `umbrella_limit` (dead) the score is GENUINELY
   unaffected; when one arrives for `age` (live) it is not. Listing both in the
   same order convinces the reader they carry equal weight.

   The warning is still PRODUCED (a user who asks "I entered something out of
   range, did the system notice?" deserves an answer), but the influential ones
   are moved to the front of the list.

   The API contract (CLAUDE.md) defines this field as a flat list of strings and
   it stays that way — influence information is already served through
   `/model-info -> feature_influence`, so the frontend can join the two lists.
   Making the ordering meaningful carries the same information without breaking
   the contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

# The types we accept as numeric in the contract. `bool` is left out DELIBERATELY:
# in Python `bool` is a subclass of `int`, so `isinstance(True, int)` is true and
# `True` would silently compare as 1. A client sending `true` for a numeric field
# would be treated as sending "1" rather than as an error.
NUMERIC_TYPES = (int, float)


class Guardrail:
    """Detects NUMERIC fields that fall outside the training range.

    Built once from the metadata (at application startup); `check()` is then
    called on every request. The ranges are NOT HARD-CODED — they are read from
    `metadata.training_ranges`, so retraining the model updates the guardrail by
    itself.
    """

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        training_ranges: Mapping[str, Mapping[str, Any]] = metadata["training_ranges"]
        influence: Mapping[str, Mapping[str, Any]] = metadata["feature_influence"]["features"]
        input_order: Sequence[str] = metadata["feature_list"]["pipeline_input_order"]

        # {field: (min, max)} — numeric fields only.
        self.numeric_bounds: dict[str, tuple[float, float]] = {
            name: (float(spec["min"]), float(spec["max"]))
            for name, spec in training_ranges.items()
            if spec["type"] == "numeric"
        }

        # Sort key: (is it uninfluential?, position in the input order).
        # Since `False < True`, influential fields come first; the second
        # component gives a deterministic, readable order within each group.
        position = {name: index for index, name in enumerate(input_order)}
        self._sort_key: dict[str, tuple[bool, int]] = {
            name: (not influence[name]["has_influence"], position[name])
            for name in self.numeric_bounds
        }

    def check(self, payload: Mapping[str, Any]) -> list[str]:
        """Names of the submitted fields that fall outside the training range.

        The returned list puts the fields the model actually uses first, then the
        dead ones; within each group, the pipeline input order applies.
        """
        flagged: list[str] = []

        for name, (minimum, maximum) in self.numeric_bounds.items():
            value = payload.get(name)
            if value is None:
                # Field was not sent -> it will be filled with the default (see 3).
                continue
            if isinstance(value, bool) or not isinstance(value, NUMERIC_TYPES):
                # A non-numeric value is not this layer's problem: Pydantic
                # already returns a 422. When the guardrail is called directly
                # (without HTTP), skipping beats silently making a wrong
                # comparison.
                continue

            try:
                numeric = float(value)
            except OverflowError:
                # A huge `int` such as `10**10000` does not fit in a float
                # (Codex F2-3). Unreachable through HTTP (Pydantic's bounds are
                # far tighter and the JSON encoder would blow up first), but this
                # module can also be called without HTTP. It is certainly outside
                # the range: flag it, do not crash.
                flagged.append(name)
                continue

            if math.isnan(numeric):
                # NaN compares False against everything; without this check it
                # would count as "inside the range" and pass SILENTLY.
                flagged.append(name)
                continue

            if numeric < minimum or numeric > maximum:
                flagged.append(name)

        flagged.sort(key=lambda name: self._sort_key[name])
        return flagged
