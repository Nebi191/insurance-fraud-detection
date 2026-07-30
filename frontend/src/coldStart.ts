/**
 * "Waking up" hint for the free-tier Hugging Face Space the backend runs on.
 *
 * WHY THIS EXISTS
 * ----------------
 * Hugging Face Spaces' free tier puts a container to sleep after 48 hours of
 * inactivity; waking it back up (on the next request) takes roughly 30-60
 * seconds. Locally, and on an already-warm Space, every request resolves in
 * well under a second. A generic "Loading…" label during that first cold
 * request looks indistinguishable from a hung/broken app — a client clicking
 * around has no way to tell "this is expected, wait" from "this is broken".
 *
 * WHY A THRESHOLD, NOT AN IMMEDIATE MESSAGE
 * ------------------------------------------
 * The hint must NEVER be visible for a normal, fast response (local dev, or a
 * warm Space) — a message that flashes on screen for 40ms and disappears reads
 * as a bug, not as reassurance. `useColdStartHint` only flips to `true` once
 * `active` has stayed `true` continuously for longer than `thresholdMs`; a
 * request that resolves before the timer fires never shows anything.
 */
import { useEffect, useState } from "react";

/**
 * Comfortably above any real local/warm-Space round trip (typically well under
 * a second, even accounting for a slow connection), and comfortably below the
 * 30-60s a cold start actually takes — so the hint appears early enough to be
 * reassuring rather than feeling like a second, later surprise.
 */
export const COLD_START_HINT_THRESHOLD_MS = 3500;

export const COLD_START_HINT_MESSAGE =
  "Waking the scoring service… this demo runs on free-tier hosting that sleeps " +
  "after inactivity. The first request can take about 30-60 seconds; every " +
  "request after that is instant.";

/**
 * Turns `true` only once `active` has stayed `true` continuously for longer
 * than `thresholdMs`, and turns `false` immediately when `active` becomes
 * `false` (whether or not the timer had already fired) — so the hint can never
 * outlive the request it describes, success or failure.
 */
export function useColdStartHint(
  active: boolean,
  thresholdMs: number = COLD_START_HINT_THRESHOLD_MS,
): boolean {
  const [hint, setHint] = useState(false);

  // Resetting `hint` the instant `active` goes false happens WHILE RENDERING
  // (comparing against the last-seen `active` stored in state), the same
  // "adjust state when a prop changes" pattern already used in
  // `ClaimForm.tsx`'s `FieldGroupSection` — not in a `useEffect`. That is what
  // keeps the request-just-settled render and the hint-disappears render the
  // SAME render: a `useEffect`-based reset would instead commit the stale
  // `true` first and only clear it in a following pass, i.e. one visible frame
  // of a hint describing a request that has already finished.
  const [trackedActive, setTrackedActive] = useState(active);
  if (trackedActive !== active) {
    setTrackedActive(active);
    if (!active) setHint(false);
  }

  useEffect(() => {
    if (!active) return;
    // Scheduling the flip inside a timer callback (not synchronously in the
    // effect body) is what keeps this outside `react-hooks/set-state-in-effect`'s
    // target: that rule is about a render committing and then immediately
    // triggering another one on mount, which cascades. A state update that
    // fires seconds later, off a timer, is an ordinary async event handler.
    const timer = window.setTimeout(() => setHint(true), thresholdMs);
    return () => window.clearTimeout(timer);
  }, [active, thresholdMs]);

  return hint;
}
