/**
 * The HUMAN-AUTHORED part of the form: labels, grouping, ordering.
 *
 * WHY THIS IS ONLY HALF-STATIC
 * ----------------------------
 * Everything NOT in this file comes from `/model-info`: the option lists, the
 * min/max ranges, the default values and — most importantly — which fields the
 * model does not actually use (`feature_influence`).
 *
 * Hard-coding that "unused" list here is forbidden: retraining the model changes
 * it, and an embedded copy would start lying silently. The badges are fed from a
 * single source of truth (the artifact).
 *
 * Labels and grouping, on the other hand, cannot come from the API — they are
 * wording and information-architecture decisions. If a field name changes in the
 * artifact, `assertFieldsCoverContract()` catches it at startup; we never render
 * a quietly incomplete form.
 */

import type { ModelInfoResponse } from "./types";

export interface FieldMeta {
  /** API field name, hyphenated exactly as the contract spells it (`capital-gains`). */
  name: string;
  label: string;
  /** Short helper text rendered under the label. */
  hint?: string;
  /** Step for numeric inputs (0.01 for the fractional premium). */
  step?: number;
}

export interface FieldGroup {
  id: string;
  title: string;
  description: string;
  fields: FieldMeta[];
}

/**
 * The group that starts EXPANDED: the fields that actually move the score.
 *
 * The selection is measured, not guessed (booster split counts): insured_hobbies
 * 353, incident_severity 173, policy_annual_premium 107, capital-gains 75,
 * auto_year 74, witnesses 37. The list is fixed here because which fields go in
 * the shop window is a presentation decision; the badges come from measurement.
 *
 * THIS DECISION CAN GO STALE: retraining the model can drop one of these six to
 * `has_influence: false` without anyone touching this file. `App.tsx` calls
 * `warnIfHighlightGroupIsStale()` on every successful `/model-info` load to
 * catch exactly that — see the function below for why it warns instead of
 * failing startup.
 */
export const HIGHLIGHT_GROUP: FieldGroup = {
  id: "highlights",
  title: "Strongest drivers of the score",
  description:
    "The six fields the model splits on most often. Anything you leave blank is " +
    "filled in with the median/mode of the training data.",
  fields: [
    { name: "insured_hobbies", label: "Hobby" },
    { name: "incident_severity", label: "Incident severity" },
    { name: "policy_annual_premium", label: "Annual premium", step: 0.01 },
    { name: "capital-gains", label: "Capital gains" },
    { name: "auto_year", label: "Vehicle model year" },
    { name: "witnesses", label: "Witnesses" },
  ],
};

/** The remaining fields, in their logical groups. */
export const FIELD_GROUPS: FieldGroup[] = [
  {
    id: "policy",
    title: "Policy & customer",
    description: "Fields describing the insurance relationship and policy terms.",
    fields: [
      { name: "months_as_customer", label: "Customer tenure", hint: "In months" },
      { name: "age", label: "Age" },
      { name: "policy_state", label: "Policy state" },
      { name: "policy_csl", label: "Combined single limit (CSL)", hint: "Per person / per occurrence" },
      { name: "policy_deductable", label: "Deductible" },
      { name: "umbrella_limit", label: "Umbrella limit" },
      { name: "policy_bind_year", label: "Policy bind year" },
    ],
  },
  {
    id: "insured",
    title: "Insured profile",
    description: "Demographic and financial details of the policyholder.",
    fields: [
      { name: "insured_sex", label: "Sex" },
      { name: "insured_education_level", label: "Education level" },
      { name: "insured_occupation", label: "Occupation" },
      { name: "insured_relationship", label: "Household relationship" },
      {
        name: "capital-loss",
        label: "Capital loss",
        hint: "Encoded as a negative number in this dataset",
      },
    ],
  },
  {
    id: "incident",
    title: "Incident",
    description: "Fields describing the accident itself.",
    fields: [
      { name: "incident_type", label: "Incident type" },
      { name: "collision_type", label: "Collision type" },
      { name: "authorities_contacted", label: "Authorities contacted" },
      { name: "incident_state", label: "Incident state" },
      { name: "incident_city", label: "Incident city" },
      { name: "incident_hour_of_the_day", label: "Hour of the day", hint: "0-23" },
      { name: "number_of_vehicles_involved", label: "Vehicles involved" },
      { name: "property_damage", label: "Property damage" },
      { name: "bodily_injuries", label: "Bodily injuries" },
      { name: "police_report_available", label: "Police report available" },
      {
        name: "incident_year",
        label: "Incident year",
        hint: "Every training row is 2015 — any other year counts as out of distribution",
      },
    ],
  },
  {
    id: "claim",
    title: "Claim amounts",
    description: "The amounts being claimed.",
    fields: [
      { name: "total_claim_amount", label: "Total claim amount" },
      { name: "injury_claim", label: "Injury claim" },
      { name: "property_claim", label: "Property claim" },
      { name: "vehicle_claim", label: "Vehicle claim" },
    ],
  },
  {
    id: "vehicle",
    title: "Vehicle",
    description: "Fields describing the insured vehicle.",
    fields: [{ name: "auto_make", label: "Vehicle make" }],
  },
];

export const ALL_GROUPS: FieldGroup[] = [HIGHLIGHT_GROUP, ...FIELD_GROUPS];

/** Every field name defined across the form. */
export const ALL_FIELD_NAMES: string[] = ALL_GROUPS.flatMap((group) =>
  group.fields.map((field) => field.name),
);

/**
 * API field name -> human-readable label.
 *
 * The SHAP chart and the warning lists arrive from the backend with raw field
 * names (`capital-gains`); reusing the same label there ties the chart back to
 * the form. The table is DERIVED from `ALL_GROUPS` — keeping a second hand-written
 * list would only guarantee the two drift apart.
 */
export const FIELD_LABELS: Record<string, string> = Object.fromEntries(
  ALL_GROUPS.flatMap((group) => group.fields.map((field) => [field.name, field.label])),
);

/** Falls back to the raw name so an unknown field never renders blank. */
export function fieldLabel(name: string): string {
  return FIELD_LABELS[name] ?? name;
}

/**
 * Display forms for categorical values.
 *
 * The value SENT to the API is always the key itself; only the rendering
 * changes. This map exists because the raw dataset shouts (`YES`, `MALE`) and
 * encodes missingness as a bare `?`. Values that already read well (hobbies,
 * occupations, makes) are left exactly as the dataset spells them.
 */
export const VALUE_LABELS: Record<string, string> = {
  "?": "Unknown (?)",
  YES: "Yes",
  NO: "No",
  MALE: "Male",
  FEMALE: "Female",
};

export function valueLabel(value: string): string {
  return VALUE_LABELS[value] ?? value;
}

/**
 * Verifies that the form covers the API contract exactly.
 *
 * WHY THIS IS NEEDED: the field list is hand-written, so it goes stale silently
 * when the artifact changes. A missing field is an input the user can never see;
 * an extra field means every request gets a 422 because of `extra="forbid"`.
 * Both must fail loudly rather than quietly.
 */
export function assertFieldsCoverContract(pipelineInputOrder: string[]): void {
  const declared = new Set(ALL_FIELD_NAMES);
  const contract = new Set(pipelineInputOrder);

  const missing = pipelineInputOrder.filter((name) => !declared.has(name));
  const extra = ALL_FIELD_NAMES.filter((name) => !contract.has(name));
  const duplicates = ALL_FIELD_NAMES.filter(
    (name, index) => ALL_FIELD_NAMES.indexOf(name) !== index,
  );

  const problems: string[] = [];
  if (missing.length) problems.push(`field(s) missing from the form: ${missing.join(", ")}`);
  if (extra.length) problems.push(`field(s) not in the API contract: ${extra.join(", ")}`);
  if (duplicates.length) problems.push(`declared twice: ${duplicates.join(", ")}`);

  if (problems.length) {
    throw new Error(
      `Form fields do not match the API contract — ${problems.join(" | ")}. ` +
        "src/fields.ts needs updating.",
    );
  }
}

/**
 * Returns the `HIGHLIGHT_GROUP` fields the artifact's MEASURED influence no
 * longer backs up (`has_influence` is `false` or the field is missing from
 * `feature_influence` entirely — both count as "not actually among the
 * effective ones").
 *
 * Pure and side-effect free on purpose: this is what a unit test asserts
 * against directly, without needing to spy on `console.warn`.
 */
export function findStaleHighlightGroupFields(
  featureInfluence: ModelInfoResponse["feature_influence"],
): FieldMeta[] {
  return HIGHLIGHT_GROUP.fields.filter(
    (field) => featureInfluence.features[field.name]?.has_influence !== true,
  );
}

/**
 * Warns — loudly, but WITHOUT stopping the app — when `HIGHLIGHT_GROUP` (see
 * above) no longer matches the artifact's measured `feature_influence`.
 *
 * WHY A WARNING AND NOT A THROWN ERROR (unlike `assertFieldsCoverContract`):
 * which six fields sit in the showcase is a presentation decision a human
 * made, not a data contract the app can enforce. Throwing here would turn a
 * routine retrain into a blank screen in front of a client watching the demo,
 * over a cosmetic staleness that a form the user can still fully use does not
 * warrant. `assertFieldsCoverContract` fails hard because its failure mode is
 * "the user cannot see or submit a field that exists" — a real, immediate
 * capability loss. This failure mode is "the six spotlighted fields are a
 * slightly outdated recommendation" — the rest of the form (every field, its
 * range, its `unused by model` badge) still renders correctly from the same
 * `/model-info` response, so there is nothing to protect the user from.
 *
 * WHY NOT LEAVE IT TO A TEST ALONE: a test only runs when someone remembers to
 * run it, typically BEFORE a retrain, not after the freshly retrained artifact
 * is dropped into `backend/models/`. A console warning fires the moment the
 * exact artifact in front of a developer disagrees with this file, including
 * during manual QA of a freshly deployed backend — which is when it is most
 * likely to be noticed AND acted on. The unit test below covers the pure
 * check's correctness; this function covers when it actually gets checked.
 */
export function warnIfHighlightGroupIsStale(
  featureInfluence: ModelInfoResponse["feature_influence"],
): void {
  const stale = findStaleHighlightGroupFields(featureInfluence);
  if (stale.length === 0) return;

  console.warn(
    "[HIGHLIGHT_GROUP stale] src/fields.ts showcases " +
      `${stale.map((field) => field.name).join(", ")} as one of the model's strongest ` +
      "drivers, but the current artifact's feature_influence says the trained trees do " +
      "not split on it (has_influence: false) — the model was likely retrained. " +
      "HIGHLIGHT_GROUP needs a human to pick new fields; this is a presentation " +
      "decision, not something the app corrects on its own.",
  );
}
