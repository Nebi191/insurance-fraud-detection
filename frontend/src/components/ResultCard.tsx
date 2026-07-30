/**
 * Score card + out-of-distribution (OOD) warning banner.
 *
 * THE PROBABILITY IS NOT A "PERCENT CHANCE"
 * -----------------------------------------
 * The model was trained with `scale_pos_weight ≈ 3.04`: the minority class was
 * weighted three times heavier, which systematically inflates `predict_proba`
 * output. So 0.73 does NOT mean "this claim is 73% likely to be fraud"; it only
 * means "ranked high relative to other claims".
 *
 * That warning sits on the card itself, not in a footnote — if an adjuster
 * misreads the number, the whole demo is misunderstood. The text comes from the
 * backend (`/model-info -> probability_calibration.warning`); the frontend does
 * not keep its own copy.
 */

import { fieldLabel } from "../fields";
import type { ModelInfoResponse, PredictResponse, RiskLevel } from "../types";
import { ShapChart } from "./ShapChart";

const RISK_PRESENTATION: Record<RiskLevel, { label: string; className: string; bar: string }> = {
  low: {
    label: "Low risk",
    className:
      "border-[var(--color-risk-low)] bg-[color-mix(in_oklch,var(--color-risk-low)_12%,transparent)] text-[var(--color-risk-low)]",
    bar: "bg-[var(--color-risk-low)]",
  },
  medium: {
    label: "Medium risk",
    className:
      "border-[var(--color-risk-medium)] bg-[color-mix(in_oklch,var(--color-risk-medium)_14%,transparent)] text-[var(--color-risk-medium)]",
    bar: "bg-[var(--color-risk-medium)]",
  },
  high: {
    label: "High risk",
    className:
      "border-[var(--color-risk-high)] bg-[color-mix(in_oklch,var(--color-risk-high)_12%,transparent)] text-[var(--color-risk-high)]",
    bar: "bg-[var(--color-risk-high)]",
  },
};

export function ResultCard({
  result,
  info,
}: {
  result: PredictResponse;
  info: ModelInfoResponse;
}) {
  const presentation = RISK_PRESENTATION[result.risk_level];
  const percentage = result.fraud_probability * 100;

  return (
    <div className="flex flex-col gap-4">
      {result.out_of_distribution_warnings.length > 0 && (
        <OodBanner fields={result.out_of_distribution_warnings} info={info} />
      )}

      <section className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-medium tracking-wide text-slate-500 uppercase dark:text-slate-400">
              Fraud score
            </p>
            <p className="mt-1 text-4xl font-bold tabular-nums text-slate-900 dark:text-slate-50">
              {percentage.toFixed(1)}
              <span className="text-2xl text-slate-400">%</span>
            </p>
          </div>
          <span
            className={`rounded-full border px-3 py-1 text-sm font-semibold ${presentation.className}`}
          >
            {presentation.label}
          </span>
        </div>

        {/* The threshold markers show where on the scale we landed; the
            thresholds come from the backend, the frontend keeps no copy. Their
            horizontal position is the ACTUAL threshold percentage (not an
            evenly-spaced guess) so the labels stay correct if the backend ever
            changes the cut points. */}
        <div className="relative mt-4 h-2.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
          <div
            className={`h-full rounded-full transition-all ${presentation.bar}`}
            style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
          />
        </div>
        <div className="relative mt-1.5 h-3.5 w-full text-[11px] text-slate-400 dark:text-slate-500">
          <span className="absolute left-0">0</span>
          <ThresholdLabel
            percent={info.risk_thresholds.low_below * 100}
            label={`medium at ${(info.risk_thresholds.low_below * 100).toFixed(0)}%`}
          />
          <ThresholdLabel
            percent={info.risk_thresholds.high_at_or_above * 100}
            label={`high at ${(info.risk_thresholds.high_at_or_above * 100).toFixed(0)}%`}
          />
          <span className="absolute right-0">100</span>
        </div>

        <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs leading-relaxed text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          <p className="font-semibold">This percentage is a ranking score, not a probability.</p>
          <p className="mt-1">{info.probability_calibration.warning}</p>
        </div>
      </section>

      <ShapChart values={result.shap_values} />
    </div>
  );
}

/**
 * Out-of-distribution input warning.
 *
 * The guardrail returns an ORDERED list: fields the model actually uses come
 * first. The banner preserves that order and additionally marks the dead fields,
 * because an "out of range" warning on `umbrella_limit` does not carry the same
 * weight as the same warning on `age` — in the first case the score genuinely
 * was not affected.
 */
function OodBanner({ fields, info }: { fields: string[]; info: ModelInfoResponse }) {
  return (
    <section
      role="status"
      className="rounded-xl border border-amber-400 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/40"
    >
      <div className="flex items-start gap-3">
        <span aria-hidden="true" className="mt-0.5 text-lg leading-none text-amber-600 dark:text-amber-500">
          ▲
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-amber-900 dark:text-amber-200">
            {fields.length} {fields.length === 1 ? "field falls" : "fields fall"} outside the range the model saw in training
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-amber-800 dark:text-amber-300">
            The score was still computed — a warning never blocks a prediction. But
            the model has never seen these values, so the number it produced for
            this claim is silent extrapolation. Tree models cannot{" "}
            <em>reach past</em> their training range: they apply what they learned
            up to the most extreme value they saw and cannot tell anything beyond
            it apart.
          </p>

          <ul className="mt-3 flex flex-wrap gap-2">
            {fields.map((name) => {
              const range = info.training_ranges[name];
              const dead = info.feature_influence.features[name]?.has_influence === false;
              return (
                <li
                  key={name}
                  className="rounded-lg border border-amber-300 bg-white px-2.5 py-1 text-xs dark:border-amber-800 dark:bg-slate-900"
                >
                  <span className="font-medium text-amber-900 dark:text-amber-200">
                    {fieldLabel(name)}
                  </span>
                  {range?.min != null && range?.max != null && (
                    <span className="ml-1.5 tabular-nums text-amber-700 dark:text-amber-400">
                      (training: {formatCompact(range.min)}–{formatCompact(range.max)})
                    </span>
                  )}
                  {dead && (
                    <span className="ml-1.5 text-slate-500 dark:text-slate-400">
                      · unused by the model, score unaffected
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </section>
  );
}

/**
 * A single threshold label, positioned at its REAL percentage along the bar
 * (`left: X%`), not spaced evenly with `flex justify-between` — the two are
 * only close by coincidence for today's 35/65 thresholds.
 *
 * Near either edge the label is anchored by its start/end instead of its
 * center, so its text stays inside the bar instead of spilling past 0%/100%.
 */
function ThresholdLabel({ percent, label }: { percent: number; label: string }) {
  const clamped = Math.min(100, Math.max(0, percent));
  const nearStart = clamped <= 12;
  const nearEnd = clamped >= 88;
  const translateX = nearStart ? "0%" : nearEnd ? "-100%" : "-50%";

  return (
    <span
      className="absolute top-0 whitespace-nowrap"
      style={{ left: `${clamped}%`, transform: `translateX(${translateX})` }}
    >
      {label}
    </span>
  );
}

function formatCompact(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: Math.abs(value) >= 100_000 ? "compact" : "standard",
    maximumFractionDigits: 2,
  }).format(value);
}
