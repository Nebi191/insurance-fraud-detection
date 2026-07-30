/**
 * Cold-start hint (Phase 7): `App` fetches `/model-info` at startup and shows
 * a plain "Loading…" state while it is in flight. These tests cover the
 * amendment described in `src/coldStart.ts` — a hint that explains a Hugging
 * Face Spaces free-tier cold start (~30-60s) must appear once the request has
 * been running past `COLD_START_HINT_THRESHOLD_MS`, and must NEVER appear for
 * a normal, fast response (local dev, or an already-warm Space).
 *
 * `vi.useFakeTimers()` drives the threshold deterministically — no real
 * `setTimeout` wait, no flakiness.
 */
import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchModelInfo } from "./api";
import App from "./App";
import { COLD_START_HINT_THRESHOLD_MS } from "./coldStart";
import { buildModelInfoFixture } from "./testUtils/modelInfoFixture";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, fetchModelInfo: vi.fn() };
});

const fetchModelInfoMock = vi.mocked(fetchModelInfo);

/** A promise this test controls the settlement of, instead of a real network delay. */
function createDeferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("App — cold-start hint while loading /model-info", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fetchModelInfoMock.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not show the hint before the threshold elapses", () => {
    const deferred = createDeferred<ReturnType<typeof buildModelInfoFixture>>();
    fetchModelInfoMock.mockReturnValueOnce(deferred.promise);

    render(<App />);

    expect(screen.getByText("Loading model info…")).toBeInTheDocument();
    expect(screen.queryByText(/Waking the scoring service/)).not.toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(COLD_START_HINT_THRESHOLD_MS - 1);
    });

    expect(screen.queryByText(/Waking the scoring service/)).not.toBeInTheDocument();
  });

  it("shows the hint once the request has run past the threshold", async () => {
    const deferred = createDeferred<ReturnType<typeof buildModelInfoFixture>>();
    fetchModelInfoMock.mockReturnValueOnce(deferred.promise);

    render(<App />);

    act(() => {
      vi.advanceTimersByTime(COLD_START_HINT_THRESHOLD_MS);
    });

    expect(screen.getByText(/Waking the scoring service/)).toBeInTheDocument();

    // Resolving the request must retire the hint immediately — it must not
    // outlive the loading state it describes.
    await act(async () => {
      deferred.resolve(buildModelInfoFixture());
      await Promise.resolve();
    });

    expect(screen.queryByText(/Waking the scoring service/)).not.toBeInTheDocument();
  });

  it("never shows the hint for a response that resolves before the threshold", async () => {
    fetchModelInfoMock.mockResolvedValueOnce(buildModelInfoFixture());

    render(<App />);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    // The page has finished loading (fast path) — advancing time afterwards
    // must not retroactively conjure the hint.
    act(() => {
      vi.advanceTimersByTime(COLD_START_HINT_THRESHOLD_MS * 2);
    });

    expect(screen.queryByText(/Waking the scoring service/)).not.toBeInTheDocument();
  });
});
