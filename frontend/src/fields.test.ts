/**
 * Unit tests for the "single source of truth" checks in `fields.ts`
 * (Faz 6 review, low-priority item 2: `HIGHLIGHT_GROUP` staleness).
 *
 * `HIGHLIGHT_GROUP` is a presentation decision (see the doc comment on it) —
 * these tests do NOT assert on which fields it contains. They only assert
 * that `findStaleHighlightGroupFields`/`warnIfHighlightGroupIsStale` correctly
 * notice when the artifact's MEASURED `feature_influence` no longer backs up
 * whatever fields happen to be in the group.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  findStaleHighlightGroupFields,
  HIGHLIGHT_GROUP,
  warnIfHighlightGroupIsStale,
  type FieldMeta,
} from "./fields";
import type { ModelInfoResponse } from "./types";

/**
 * `HIGHLIGHT_GROUP.fields[0]` typechecks as possibly `undefined`
 * (`noUncheckedIndexedAccess`) even though the group is never empty in
 * practice. This makes that assumption explicit instead of silencing the
 * checker with a non-null assertion.
 */
function firstHighlightField(): FieldMeta {
  const [first] = HIGHLIGHT_GROUP.fields;
  if (!first) throw new Error("HIGHLIGHT_GROUP.fields must not be empty");
  return first;
}

/** A `feature_influence.features` map where every `HIGHLIGHT_GROUP` field has influence. */
function influentialFeatureInfluence(): ModelInfoResponse["feature_influence"] {
  const features: ModelInfoResponse["feature_influence"]["features"] = {};
  for (const field of HIGHLIGHT_GROUP.fields) {
    features[field.name] = { split_count: 42, gain: 1, has_influence: true };
  }
  return {
    description: "test fixture",
    measured_from: "test fixture",
    interpretation_note: "test fixture",
    summary: {
      n_features: HIGHLIGHT_GROUP.fields.length,
      n_with_influence: HIGHLIGHT_GROUP.fields.length,
      n_without_influence: 0,
      features_without_influence: [],
    },
    features,
  };
}

describe("findStaleHighlightGroupFields", () => {
  it("returns nothing when every showcased field still has influence", () => {
    expect(findStaleHighlightGroupFields(influentialFeatureInfluence())).toEqual([]);
  });

  it("flags a showcased field the retrained booster no longer splits on", () => {
    const featureInfluence = influentialFeatureInfluence();
    const first = firstHighlightField();
    featureInfluence.features[first.name] = { split_count: 0, gain: 0, has_influence: false };

    const stale = findStaleHighlightGroupFields(featureInfluence);

    expect(stale).toEqual([first]);
  });

  it("flags a showcased field missing from feature_influence entirely", () => {
    const featureInfluence = influentialFeatureInfluence();
    const first = firstHighlightField();
    delete featureInfluence.features[first.name];

    const stale = findStaleHighlightGroupFields(featureInfluence);

    expect(stale).toEqual([first]);
  });

  it("flags every showcased field that lost influence, not just the first", () => {
    const featureInfluence = influentialFeatureInfluence();
    for (const field of HIGHLIGHT_GROUP.fields) {
      featureInfluence.features[field.name] = { split_count: 0, gain: 0, has_influence: false };
    }

    const stale = findStaleHighlightGroupFields(featureInfluence);

    expect(stale).toEqual(HIGHLIGHT_GROUP.fields);
  });
});

describe("warnIfHighlightGroupIsStale", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("stays silent when the showcase group matches measured influence", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    warnIfHighlightGroupIsStale(influentialFeatureInfluence());

    expect(warn).not.toHaveBeenCalled();
  });

  it("warns, naming the field, when a showcased field has gone stale", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const featureInfluence = influentialFeatureInfluence();
    const first = firstHighlightField();
    featureInfluence.features[first.name] = { split_count: 0, gain: 0, has_influence: false };

    warnIfHighlightGroupIsStale(featureInfluence);

    expect(warn).toHaveBeenCalledTimes(1);
    const [message] = warn.mock.calls[0] as [string];
    expect(message).toContain("HIGHLIGHT_GROUP stale");
    expect(message).toContain(first.name);
  });
});
