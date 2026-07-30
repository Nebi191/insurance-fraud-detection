/**
 * Regression tests for the request-generation race guard (Phase 6 review,
 * items A1-A3).
 *
 * `predict()` is mocked with a DEFERRED promise so the test controls exactly
 * when the "network" resolves/rejects — no `setTimeout`, no flakiness.
 *
 * `ResultCard` is mocked to a trivial stub: these tests exercise `ScorePage`'s
 * own state orchestration (which response gets applied, when), not the SHAP
 * chart or OOD banner rendering, which belong to `ResultCard`'s own concerns.
 *
 * Item B (the non-finite-number guard) is NOT tested here through the DOM: in
 * jsdom, `<input type="number">` sanitizes a value like "1e999" down to the
 * empty string at the DOM layer itself (confirmed by direct experiment —
 * `input.value = "1e999"` already reads back as `""`, before React ever sees
 * it). Real browsers keep the literal string in that case, which is exactly
 * what makes the underlying bug reachable there. Because jsdom cannot
 * reproduce that browser behaviour, Item B is covered by a direct unit test
 * of `buildPayload` instead — see `src/components/ClaimForm.test.tsx`, which
 * exercises the exact function the fix lives in without going through a
 * `<input type="number">` element at all.
 */
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ApiError, predict } from "../api";
import { buildModelInfoFixture, buildPredictResponseFixture } from "../testUtils/modelInfoFixture";
import type { PredictResponse } from "../types";
import { ScorePage } from "./ScorePage";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, predict: vi.fn() };
});

vi.mock("../components/ResultCard", () => ({
  ResultCard: ({ result }: { result: PredictResponse }) => (
    <div data-testid="result-card">{result.fraud_probability}</div>
  ),
}));

const predictMock = vi.mocked(predict);

/** A promise this test controls the settlement of, instead of a real network delay. */
function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Drains the microtask queue a fixed number of times after a deferred settles,
 *  so the component's own `await predict(...)` continuation has definitely run
 *  before we assert. This is deterministic (no timers, no arbitrary delay) —
 *  it just lets already-scheduled microtasks flush. */
async function flushMicrotasks(times = 5): Promise<void> {
  for (let i = 0; i < times; i += 1) {
    await Promise.resolve();
  }
}

describe("ScorePage — request generation race guard", () => {
  it("A1: an in-flight request's response does not mark a since-edited result as fresh", async () => {
    predictMock.mockReset();
    const user = userEvent.setup();
    const info = buildModelInfoFixture();
    const deferred = createDeferred<PredictResponse>();
    predictMock.mockReturnValueOnce(deferred.promise);

    render(<ScorePage info={info} />);

    const witnesses = screen.getByLabelText("Witnesses");
    await user.type(witnesses, "2");
    await user.click(screen.getByRole("button", { name: "Score claim" }));

    // Edit a field while the request is still in flight.
    await user.clear(witnesses);
    await user.type(witnesses, "9");

    await act(async () => {
      deferred.resolve(buildPredictResponseFixture());
      await flushMicrotasks();
    });

    // The response DOES get applied (it is still the answer to what was
    // asked) — but it must be marked stale, not fresh, because the form has
    // since changed.
    expect(await screen.findByTestId("result-card")).toBeInTheDocument();
    expect(
      screen.getByText(/The form changed — the result below belongs to the previous input/),
    ).toBeInTheDocument();
  });

  it("A2: a request that resolves after 'Clear form' does not paint a ghost result card", async () => {
    predictMock.mockReset();
    const user = userEvent.setup();
    const info = buildModelInfoFixture();
    const deferred = createDeferred<PredictResponse>();
    predictMock.mockReturnValueOnce(deferred.promise);

    render(<ScorePage info={info} />);

    await user.type(screen.getByLabelText("Witnesses"), "2");
    await user.click(screen.getByRole("button", { name: "Score claim" }));

    await user.click(screen.getByRole("button", { name: "Clear form" }));
    expect(screen.getByText("The result will appear here")).toBeInTheDocument();

    await act(async () => {
      deferred.resolve(buildPredictResponseFixture());
      await flushMicrotasks();
    });

    expect(screen.queryByTestId("result-card")).not.toBeInTheDocument();
    expect(screen.getByText("The result will appear here")).toBeInTheDocument();
  });

  it("A3: a validation error from a request invalidated by 'Clear form' does not paint the cleared form", async () => {
    predictMock.mockReset();
    const user = userEvent.setup();
    const info = buildModelInfoFixture();
    const deferred = createDeferred<PredictResponse>();
    predictMock.mockReturnValueOnce(deferred.promise);

    render(<ScorePage info={info} />);

    await user.type(screen.getByLabelText("Witnesses"), "2");
    await user.click(screen.getByRole("button", { name: "Score claim" }));

    await user.click(screen.getByRole("button", { name: "Clear form" }));

    await act(async () => {
      deferred.reject(
        new ApiError("validation", "The input did not pass validation.", 422, {
          witnesses: "must be less than or equal to 10",
        }),
      );
      await flushMicrotasks();
    });

    expect(screen.queryByText(/must be less than or equal to 10/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Some fields were rejected/)).not.toBeInTheDocument();
    expect(screen.queryByText(/did not pass validation/)).not.toBeInTheDocument();
    expect(screen.getByLabelText<HTMLInputElement>("Witnesses").value).toBe("");
  });
});
