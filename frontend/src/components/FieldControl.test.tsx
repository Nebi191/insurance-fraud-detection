/**
 * Regression test for C3 (Phase 6 review): the form must show BOTH the
 * training range and the API's hard physical bound, distinguishable, and
 * must not fabricate a physical bound for a categorical field.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { FieldMeta } from "../fields";
import type { DefaultInfo, PhysicalRange, TrainingRange } from "../types";
import { FieldControl } from "./FieldControl";

const meta: FieldMeta = { name: "witnesses", label: "Witnesses" };
const fallback: DefaultInfo = { value: 1, dtype: "int64", strategy: "median" };

describe("FieldControl — training range vs. physical range", () => {
  it("shows both ranges, distinguishable, for a numeric field with a physical bound", () => {
    const range: TrainingRange = { type: "numeric", dtype: "int64", min: 0, max: 3 };
    const physicalRange: PhysicalRange = { min: 0, max: 10 };

    render(
      <FieldControl
        meta={meta}
        range={range}
        physicalRange={physicalRange}
        fallback={fallback}
        splitCount={37}
        hasInfluence={true}
        value=""
        onChange={() => {}}
      />,
    );

    expect(screen.getByText(/Seen in training: 0 – 3/)).toBeInTheDocument();
    expect(screen.getByText(/Accepted: 0 – 10/)).toBeInTheDocument();
  });

  it("shows only the existing text for a categorical field (no physical range fabricated)", () => {
    const range: TrainingRange = {
      type: "categorical",
      dtype: "object",
      categories: ["YES", "NO"],
    };

    render(
      <FieldControl
        meta={{ name: "police_report_available", label: "Police report available" }}
        range={range}
        physicalRange={undefined}
        fallback={{ value: "NO", dtype: "object", strategy: "mode" }}
        splitCount={0}
        hasInfluence={false}
        value=""
        onChange={() => {}}
      />,
    );

    expect(screen.queryByText(/Seen in training/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Accepted/)).not.toBeInTheDocument();
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });
});
