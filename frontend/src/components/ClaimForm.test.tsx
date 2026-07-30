/**
 * Unit tests for `buildPayload` (Phase 6 review, item B).
 *
 * WHY A UNIT TEST, NOT A DOM/INTEGRATION TEST
 * --------------------------------------------
 * The bug is: typing "1e999" into `<input type="number">` produces a string
 * that `Number(...)` parses to `Infinity`, and the OLD code silently dropped
 * that field instead of reporting it. In a real browser the input keeps the
 * literal string "1e999" (its sanitization is purely syntactic, not
 * magnitude-aware), which is what makes the bug reachable there.
 *
 * jsdom's `<input type="number">` does not behave the same way: direct
 * experiment confirmed that setting `input.value = "1e999"` (with no React or
 * event dispatch involved at all) already reads back as `""` inside jsdom —
 * the value never survives long enough to reach `buildPayload` through any
 * DOM interaction (typing, pasting, or a raw `fireEvent.change`). Simulating
 * the browser behaviour that triggers this bug is therefore not possible in
 * this test environment, so this suite calls `buildPayload` directly instead,
 * exercising exactly the function the fix lives in.
 */
import { describe, expect, it } from "vitest";

import { buildModelInfoFixture } from "../testUtils/modelInfoFixture";
import { buildPayload } from "./ClaimForm";

describe("buildPayload", () => {
  const info = buildModelInfoFixture(); // every field is declared numeric in this fixture

  it("packs a finite numeric value into the payload", () => {
    const { payload, fieldErrors } = buildPayload({ witnesses: "2" }, info);
    expect(payload).toEqual({ witnesses: 2 });
    expect(fieldErrors).toEqual({});
  });

  it("rejects a value that parses to Infinity instead of silently dropping it", () => {
    const { payload, fieldErrors } = buildPayload({ witnesses: "1e999" }, info);
    expect(payload).toEqual({});
    expect(fieldErrors).toEqual({ witnesses: "Enter a finite number." });
  });

  it("rejects a value that parses to -Infinity", () => {
    const { payload, fieldErrors } = buildPayload({ witnesses: "-1e999" }, info);
    expect(payload).toEqual({});
    expect(fieldErrors).toEqual({ witnesses: "Enter a finite number." });
  });

  it("a rejected field does not silently look the same as an untouched one", () => {
    // The whole point of the fix: an untouched field and a rejected one must
    // be distinguishable, so the caller (`ScorePage`) can block the request
    // and show an error rather than send a request that quietly used the
    // server-side default.
    const untouched = buildPayload({}, info);
    const rejected = buildPayload({ witnesses: "1e999" }, info);

    expect(untouched.payload).toEqual({});
    expect(untouched.fieldErrors).toEqual({});
    expect(rejected.payload).toEqual({});
    expect(rejected.fieldErrors).not.toEqual({});
  });

  it("still treats a blank field as 'do not send', not an error", () => {
    const { payload, fieldErrors } = buildPayload({ witnesses: "   " }, info);
    expect(payload).toEqual({});
    expect(fieldErrors).toEqual({});
  });

  it("passes categorical values through as strings, untouched", () => {
    const categoricalInfo = buildModelInfoFixture();
    categoricalInfo.training_ranges["incident_severity"] = {
      type: "categorical",
      dtype: "object",
      categories: ["Minor Damage", "Major Damage"],
    };
    const { payload, fieldErrors } = buildPayload(
      { incident_severity: "Major Damage" },
      categoricalInfo,
    );
    expect(payload).toEqual({ incident_severity: "Major Damage" });
    expect(fieldErrors).toEqual({});
  });
});
