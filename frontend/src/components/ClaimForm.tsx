/**
 * The claim form — 34 fields, every one of them optional.
 *
 * LAYOUT DECISION
 * ---------------
 * The six fields that actually move the score sit in an EXPANDED group; the rest
 * are collapsed. No field is hidden — the group headers also state how many
 * fields they hold and how many of those the model ignores, because
 * "Incident (11 fields, 9 unused)" is a finding in its own right.
 *
 * BLANK MEANS NOT SENT
 * `buildPayload` only packs fields that were filled in. The backend fills the
 * rest with the training median/mode, and the guardrail only checks fields that
 * were actually SENT — that is why an empty form does not produce 34 warnings.
 */

import { useState } from "react";

import { ALL_GROUPS, HIGHLIGHT_GROUP, type FieldGroup } from "../fields";
import type { ModelInfoResponse, PredictRequest } from "../types";
import { FieldControl } from "./FieldControl";

export type FormValues = Record<string, string>;

export function buildPayload(values: FormValues, info: ModelInfoResponse): PredictRequest {
  const payload: PredictRequest = {};
  for (const [name, raw] of Object.entries(values)) {
    const value = raw.trim();
    if (value === "") continue; // blank = don't send, let the default apply

    if (info.training_ranges[name]?.type === "numeric") {
      const parsed = Number(value);
      // We never send NaN: the backend would return a 422 and the user would
      // learn about the mistake from an opaque server message. The number input
      // already rejects letters; this is a second safety net.
      if (Number.isFinite(parsed)) payload[name] = parsed;
    } else {
      payload[name] = value;
    }
  }
  return payload;
}

interface ClaimFormProps {
  info: ModelInfoResponse;
  values: FormValues;
  onChange: (name: string, value: string) => void;
  onSubmit: () => void;
  onReset: () => void;
  fieldErrors: Record<string, string>;
  oodFields: Set<string>;
  submitting: boolean;
}

export function ClaimForm({
  info,
  values,
  onChange,
  onSubmit,
  onReset,
  fieldErrors,
  oodFields,
  submitting,
}: ClaimFormProps) {
  const filledCount = Object.values(values).filter((value) => value.trim() !== "").length;

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="flex flex-col gap-4"
    >
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300">
        <p>
          <strong className="font-semibold text-slate-800 dark:text-slate-100">
            Every field is optional.
          </strong>{" "}
          Anything you leave blank is filled in with the median/mode of the training
          data — you can get a score without entering anything at all.
        </p>
        <p className="mt-2">
          <span className="font-medium">{info.feature_influence.summary.n_without_influence}</span>{" "}
          fields carry an <em>unused by model</em> badge: this trained model's trees
          never branch on those columns, so their values cannot affect the score.
        </p>
      </div>

      {ALL_GROUPS.map((group) => (
        <FieldGroupSection
          key={group.id}
          group={group}
          info={info}
          values={values}
          onChange={onChange}
          fieldErrors={fieldErrors}
          oodFields={oodFields}
          defaultOpen={group.id === HIGHLIGHT_GROUP.id}
        />
      ))}

      <div className="sticky bottom-0 -mx-1 flex flex-wrap items-center gap-3 border-t border-slate-200 bg-slate-50/95 px-1 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-950/95">
        <button
          type="submit"
          disabled={submitting}
          className="rounded-lg bg-sky-700 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-sky-600 dark:hover:bg-sky-500"
        >
          {submitting ? "Scoring…" : "Score claim"}
        </button>
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
        >
          Clear form
        </button>
        <span className="text-xs text-slate-500 dark:text-slate-400">
          {filledCount === 0
            ? "No fields filled in — all 34 will use their defaults."
            : `${filledCount} filled in, the remaining ${34 - filledCount} will use their defaults.`}
        </span>
      </div>
    </form>
  );
}

function FieldGroupSection({
  group,
  info,
  values,
  onChange,
  fieldErrors,
  oodFields,
  defaultOpen,
}: {
  group: FieldGroup;
  info: ModelInfoResponse;
  values: FormValues;
  onChange: (name: string, value: string) => void;
  fieldErrors: Record<string, string>;
  oodFields: Set<string>;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  const deadCount = group.fields.filter(
    (field) => info.feature_influence.features[field.name]?.has_influence === false,
  ).length;
  const errorCount = group.fields.filter((field) => fieldErrors[field.name]).length;
  const oodCount = group.fields.filter((field) => oodFields.has(field.name)).length;
  const sectionId = `group-${group.id}`;

  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <h2>
        <button
          type="button"
          onClick={() => setOpen((previous) => !previous)}
          aria-expanded={open}
          aria-controls={sectionId}
          className="flex w-full items-center gap-3 px-4 py-3 text-left transition hover:bg-slate-50 dark:hover:bg-slate-800/60"
        >
          <span
            aria-hidden="true"
            className={`text-slate-400 transition-transform ${open ? "rotate-90" : ""}`}
          >
            ▶
          </span>
          <span className="flex-1">
            <span className="block text-sm font-semibold text-slate-800 dark:text-slate-100">
              {group.title}
            </span>
            <span className="block text-xs text-slate-500 dark:text-slate-400">
              {group.fields.length} {group.fields.length === 1 ? "field" : "fields"}
              {deadCount > 0 && `, ${deadCount} unused by the model`}
            </span>
          </span>
          {errorCount > 0 && (
            <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-semibold text-red-700 dark:bg-red-950 dark:text-red-300">
              {errorCount} {errorCount === 1 ? "error" : "errors"}
            </span>
          )}
          {oodCount > 0 && (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800 dark:bg-amber-950 dark:text-amber-300">
              {oodCount} out of range
            </span>
          )}
        </button>
      </h2>

      {open && (
        <div id={sectionId} className="border-t border-slate-200 px-4 py-4 dark:border-slate-800">
          <p className="mb-4 text-xs text-slate-500 dark:text-slate-400">{group.description}</p>
          <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2">
            {group.fields.map((field) => {
              const range = info.training_ranges[field.name];
              const fallback = info.defaults[field.name];
              const influence = info.feature_influence.features[field.name];
              // Contract coverage is verified at startup; even so we stay
              // type-safe here rather than silently skipping a missing field.
              if (!range || !fallback || !influence) return null;

              return (
                <FieldControl
                  key={field.name}
                  meta={field}
                  range={range}
                  fallback={fallback}
                  splitCount={influence.split_count}
                  hasInfluence={influence.has_influence}
                  value={values[field.name] ?? ""}
                  onChange={(value) => onChange(field.name, value)}
                  error={fieldErrors[field.name]}
                  outOfDistribution={oodFields.has(field.name)}
                />
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
