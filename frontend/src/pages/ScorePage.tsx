import { Suspense, lazy, useCallback, useMemo, useState } from "react";

import { ApiError, predict } from "../api";
import { ClaimForm, buildPayload, type FormValues } from "../components/ClaimForm";
import type { ModelInfoResponse, PredictResponse } from "../types";

/**
 * The result card is LAZY-loaded because the `ShapChart` inside it depends on
 * recharts, and recharts alone is more than half the bundle. There is no point
 * downloading a charting library before the user presses "Score claim": the
 * initial load gets noticeably faster, at the cost of a small delay once a result
 * arrives (during which we are waiting on the network anyway).
 */
const ResultCard = lazy(() =>
  import("../components/ResultCard").then((module) => ({ default: module.ResultCard })),
);

/**
 * Scoring page: form on the left, result on the right.
 *
 * A RESULT IS MARKED STALE, NOT DISCARDED
 * ---------------------------------------
 * When the user edits the form we do NOT leave the previous result standing as
 * if nothing happened — but we do not delete it either; we mark it "stale".
 * Leaving it silently would invite reading it as belonging to the edited input;
 * removing it entirely would destroy the ability to compare.
 */
export function ScorePage({ info }: { info: ModelInfoResponse }) {
  const [values, setValues] = useState<FormValues>({});
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [stale, setStale] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const oodFields = useMemo(
    () => new Set(stale ? [] : (result?.out_of_distribution_warnings ?? [])),
    [result, stale],
  );

  const handleChange = useCallback((name: string, value: string) => {
    setValues((previous) => ({ ...previous, [name]: value }));
    setStale(true);
    // Touching a field clears its error; if it is still invalid the server will
    // say so again on the next submission.
    setFieldErrors((previous) => {
      if (!(name in previous)) return previous;
      const next = { ...previous };
      delete next[name];
      return next;
    });
  }, []);

  const handleSubmit = useCallback(async () => {
    setSubmitting(true);
    setFormError(null);
    try {
      const response = await predict(buildPayload(values, info));
      setResult(response);
      setStale(false);
      setFieldErrors({});
    } catch (error) {
      if (error instanceof ApiError && error.kind === "validation") {
        setFieldErrors(error.fieldErrors);
        setFormError(
          Object.keys(error.fieldErrors).length > 0
            ? "Some fields were rejected — details are shown under each one."
            : "The input did not pass validation.",
        );
      } else {
        setFormError(
          error instanceof ApiError ? error.message : "An unexpected error occurred.",
        );
      }
    } finally {
      setSubmitting(false);
    }
  }, [values, info]);

  const handleReset = useCallback(() => {
    setValues({});
    setFieldErrors({});
    setFormError(null);
    setResult(null);
    setStale(false);
  }, []);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)] lg:items-start">
      <div className="min-w-0">
        {formError && (
          <div
            role="alert"
            className="mb-4 rounded-xl border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200"
          >
            {formError}
          </div>
        )}
        <ClaimForm
          info={info}
          values={values}
          onChange={handleChange}
          onSubmit={handleSubmit}
          onReset={handleReset}
          fieldErrors={fieldErrors}
          oodFields={oodFields}
          submitting={submitting}
        />
      </div>

      <div className="min-w-0 lg:sticky lg:top-6">
        {result === null ? (
          <EmptyState />
        ) : (
          <div className={stale ? "opacity-60 transition-opacity" : "transition-opacity"}>
            {stale && (
              <p className="mb-2 rounded-lg border border-slate-300 bg-slate-100 px-3 py-2 text-xs text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
                The form changed — the result below belongs to the previous input.
                Score it again.
              </p>
            )}
            <Suspense
              fallback={
                <div
                  aria-busy="true"
                  className="h-96 animate-pulse rounded-xl bg-slate-200 dark:bg-slate-800"
                />
              }
            >
              <ResultCard result={result} info={info} />
            </Suspense>
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <section className="rounded-xl border border-dashed border-slate-300 bg-white/50 p-8 text-center dark:border-slate-700 dark:bg-slate-900/50">
      <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
        The result will appear here
      </p>
      <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Fill in the form and press <strong>Score claim</strong>. You can also try it
        without entering anything — the model completes missing fields with the
        median/mode of the training data.
      </p>
    </section>
  );
}
