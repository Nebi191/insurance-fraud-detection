import { Suspense, lazy, useCallback, useMemo, useRef, useState } from "react";

import { ApiError, predict } from "../api";
import { ClaimForm, buildPayload, type FormValues } from "../components/ClaimForm";
import { COLD_START_HINT_MESSAGE, useColdStartHint } from "../coldStart";
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
 *
 * WHY A REQUEST GENERATION GUARD, NOT DISABLED FIELDS
 * -----------------------------------------------------
 * Fields stay editable and "Clear form" stays clickable while a `/predict`
 * request is in flight — disabling them would punish someone on a slow
 * connection by locking the form for the whole round trip. That means a
 * request can be superseded by a form edit or a reset before it resolves, in
 * two different ways that both need guarding:
 *
 * 1. RESET/RE-SUBMIT while the OLD request is still in flight: `requestIdRef`
 *    is bumped on every submit and on every reset; a response is only applied
 *    to state (`setResult`/`setStale`/`setFieldErrors`/`setSubmitting`) if it
 *    still matches the id that was current when it was sent. Both the success
 *    path and the error path check this, and so does `finally` — otherwise a
 *    slow, superseded request could flip `submitting` back to false while a
 *    newer one is in flight, or paint an unrelated/late 422 onto a form the
 *    user has since changed or cleared.
 * 2. EDIT (not reset/re-submit) while the SAME request is still in flight:
 *    `requestIdRef` does not change here, so the id check above lets the
 *    response through — correctly, it is still the answer to the question the
 *    user asked. What must NOT happen is the success handler unconditionally
 *    marking that answer "fresh": if a field was touched after the request
 *    was sent, the response no longer describes what is on screen.
 *    `editedSinceRequestRef` tracks exactly that (reset to false when a
 *    request starts, set to true by every `handleChange`), and the success
 *    handler restores `stale` from it instead of hard-coding `false`.
 */
export function ScorePage({ info }: { info: ModelInfoResponse }) {
  const [values, setValues] = useState<FormValues>({});
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [stale, setStale] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  // See `../coldStart.ts`. The very first `/predict` after the page loads is
  // already covered by `App`'s `/model-info` cold-start hint in the common
  // case (that request wakes the Space, and `ScorePage` only mounts once it
  // has answered) — but a tab left open past the Space's 48h inactivity
  // window can still hit a re-slept container on submit, so the same
  // threshold-gated hint applies here too, independently.
  const showColdStartHint = useColdStartHint(submitting);
  /** Bumped by every submit attempt so a closed field group can re-open when a
   *  fresh submission produces an error inside it (see `FieldGroupSection`). */
  const [attempt, setAttempt] = useState(0);
  /** Identifies the most recently issued `/predict` request; see the comment above. */
  const requestIdRef = useRef(0);
  /** Has any field been touched since the in-flight request was sent? See point 2 above. */
  const editedSinceRequestRef = useRef(false);

  const oodFields = useMemo(
    () => new Set(stale ? [] : (result?.out_of_distribution_warnings ?? [])),
    [result, stale],
  );

  const handleChange = useCallback((name: string, value: string) => {
    setValues((previous) => ({ ...previous, [name]: value }));
    setStale(true);
    editedSinceRequestRef.current = true;
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
    setAttempt((previous) => previous + 1);

    const { payload, fieldErrors: buildErrors } = buildPayload(values, info);
    if (Object.keys(buildErrors).length > 0) {
      // Invalid values (e.g. a non-finite number) never leave the browser: no
      // request is sent, the field is flagged the same way a 422 would flag it.
      setFieldErrors((previous) => ({ ...previous, ...buildErrors }));
      setFormError("Some fields were rejected — details are shown under each one.");
      return;
    }

    // Captured once per submit; a response is only applied to state if this id
    // is still the current one when it resolves (see the class doc-comment).
    const requestId = ++requestIdRef.current;
    editedSinceRequestRef.current = false;
    setSubmitting(true);
    setFormError(null);
    try {
      const response = await predict(payload);
      if (requestIdRef.current !== requestId) return; // superseded by a newer submit or a reset
      setResult(response);
      // Not hard-coded to `false`: if the form was edited while this request
      // was in flight, the answer we just received is already stale.
      setStale(editedSinceRequestRef.current);
      setFieldErrors({});
    } catch (error) {
      if (requestIdRef.current !== requestId) return; // ditto — a stale error must not paint the current form
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
      if (requestIdRef.current === requestId) setSubmitting(false);
    }
  }, [values, info]);

  const handleReset = useCallback(() => {
    // Invalidates any in-flight request: when it resolves, its id will no
    // longer match `requestIdRef.current` and its callbacks will no-op.
    requestIdRef.current += 1;
    editedSinceRequestRef.current = false;
    setValues({});
    setFieldErrors({});
    setFormError(null);
    setResult(null);
    setStale(false);
    setSubmitting(false);
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
          // `handleSubmit` swallows every error internally (try/catch) and
          // never rejects, so passing it here cannot produce an unhandled
          // rejection.
          // eslint-disable-next-line @typescript-eslint/no-misused-promises
          onSubmit={handleSubmit}
          onReset={handleReset}
          fieldErrors={fieldErrors}
          oodFields={oodFields}
          submitting={submitting}
          coldStartHint={showColdStartHint ? COLD_START_HINT_MESSAGE : null}
          attempt={attempt}
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
