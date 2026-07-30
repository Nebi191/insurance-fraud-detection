/**
 * A single form field.
 *
 * The input type is NOT hard-coded; it is derived from
 * `/model-info -> training_ranges`: `categorical` renders a dropdown (options are
 * the categories seen during training), `numeric` renders a number box. That way
 * the form corrects itself when the artifact changes.
 *
 * A field left blank is NOT SENT in the request at all; the backend fills it
 * with the training median/mode. That is why we show the default value as the
 * placeholder: the user never has to guess what a field they skipped was filled
 * in with.
 *
 * TWO DIFFERENT RANGES, SHOWN SEPARATELY
 * ---------------------------------------
 * "Seen in training" (`training_ranges`) and "Accepted" (`physical_ranges`)
 * answer different questions and are NOT the same number:
 *   - "Seen in training" is what the model actually saw; anything outside it
 *     still gets a 200 plus an out-of-distribution warning (silent
 *     extrapolation, not rejected).
 *   - "Accepted" is the API's hard `ge`/`le` bound; anything outside it is a
 *     422, the request never reaches the model at all.
 * `witnesses` is the case that motivates showing both: trained on 0-3, but
 * the API accepts up to 10. Entering `-1` is a 422 (outside "Accepted");
 * entering `9` is a 200 with a warning (inside "Accepted", outside "Seen in
 * training"). Neither number is hard-coded here — both come from
 * `/model-info`, and `physical_ranges` simply has no entry for a categorical
 * field, so the "Accepted" half is omitted rather than guessed.
 */

import type { DefaultInfo, PhysicalRange, TrainingRange } from "../types";
import { valueLabel, type FieldMeta } from "../fields";
import { InfluenceBadge } from "./InfluenceBadge";

export interface FieldControlProps {
  meta: FieldMeta;
  range: TrainingRange;
  /** The API's hard `ge`/`le` bound for this field; absent for categorical fields. */
  physicalRange?: PhysicalRange | undefined;
  fallback: DefaultInfo;
  splitCount: number;
  hasInfluence: boolean;
  value: string;
  onChange: (value: string) => void;
  /** Per-field 422 message from the backend. */
  error?: string | undefined;
  /** Was this field flagged out of distribution on the last prediction? */
  outOfDistribution?: boolean;
}

export function FieldControl({
  meta,
  range,
  physicalRange,
  fallback,
  splitCount,
  hasInfluence,
  value,
  onChange,
  error,
  outOfDistribution,
}: FieldControlProps) {
  const inputId = `field-${meta.name}`;
  const describedBy: string[] = [];
  if (meta.hint) describedBy.push(`${inputId}-hint`);
  if (error) describedBy.push(`${inputId}-error`);
  if (outOfDistribution) describedBy.push(`${inputId}-ood`);

  const baseInputClass = [
    "w-full rounded-lg border px-3 py-2 text-sm transition",
    "bg-white text-slate-900 placeholder:text-slate-400",
    "dark:bg-slate-900 dark:text-slate-100 dark:placeholder:text-slate-500",
    error
      ? "border-red-500 dark:border-red-500"
      : outOfDistribution
        ? "border-amber-500 dark:border-amber-500"
        : "border-slate-300 dark:border-slate-700",
  ].join(" ");

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor={inputId} className="text-sm font-medium text-slate-700 dark:text-slate-200">
          {meta.label}
        </label>
        {!hasInfluence && <InfluenceBadge splitCount={splitCount} />}
      </div>

      {meta.hint && (
        <p id={`${inputId}-hint`} className="text-xs text-slate-500 dark:text-slate-400">
          {meta.hint}
        </p>
      )}

      {range.type === "categorical" ? (
        <select
          id={inputId}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className={baseInputClass}
          aria-describedby={describedBy.join(" ") || undefined}
          aria-invalid={error ? true : undefined}
        >
          {/* The empty option means "don't send it, use the default". */}
          <option value="">— default: {valueLabel(String(fallback.value))} —</option>
          {(range.categories ?? []).map((option) => (
            <option key={option} value={option}>
              {valueLabel(option)}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={inputId}
          type="number"
          inputMode="decimal"
          step={meta.step ?? 1}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={`default: ${fallback.value}`}
          className={baseInputClass}
          aria-describedby={describedBy.join(" ") || undefined}
          aria-invalid={error ? true : undefined}
        />
      )}

      {range.type === "numeric" && range.min != null && range.max != null && (
        <p className="text-xs text-slate-400 dark:text-slate-500">
          Seen in training: {formatNumber(range.min)} – {formatNumber(range.max)}
          {physicalRange && (
            <>
              {" "}
              · Accepted: {formatNumber(physicalRange.min)} – {formatNumber(physicalRange.max)}
            </>
          )}
        </p>
      )}

      {outOfDistribution && (
        <p
          id={`${inputId}-ood`}
          className="flex items-start gap-1.5 text-xs font-medium text-amber-700 dark:text-amber-500"
        >
          <span aria-hidden="true">▲</span>
          <span>
            The model never saw this value during training. The score was still
            computed, but read it cautiously for this field.
          </span>
        </p>
      )}

      {error && (
        <p id={`${inputId}-error`} role="alert" className="text-xs font-medium text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}
