/**
 * Application shell: loads the model info once, routes between the two pages.
 *
 * `/model-info` IS FETCHED AT STARTUP
 * -----------------------------------
 * The form builds its option lists and ranges from that response, so there is
 * nothing meaningful to render until it arrives. While waiting we show a
 * skeleton; on failure we show the REASON alongside the error — "something went
 * wrong" tells a developer who forgot to start the backend absolutely nothing.
 */

import { useCallback, useEffect, useState } from "react";

import { API_BASE_URL, ApiError, fetchModelInfo } from "./api";
import { COLD_START_HINT_MESSAGE, useColdStartHint } from "./coldStart";
import { assertFieldsCoverContract } from "./fields";
import { ModelCardPage } from "./pages/ModelCardPage";
import { ScorePage } from "./pages/ScorePage";
import { RouteLink, useRoute } from "./router";
import { ThemeToggle, useTheme } from "./theme";
import type { ModelInfoResponse } from "./types";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; info: ModelInfoResponse }
  | { status: "error"; message: string };

export default function App() {
  const { theme, toggle } = useTheme();
  const { route, navigate } = useRoute();
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const showColdStartHint = useColdStartHint(state.status === "loading");

  const load = useCallback(async () => {
    setState({ status: "loading" });
    try {
      const info = await fetchModelInfo();
      // If the hand-written field list disagrees with the artifact we do NOT stay
      // quiet: a missing field is an input the user can never see, and an extra
      // field means every request gets a 422 because of `extra="forbid"`.
      assertFieldsCoverContract(info.feature_list.pipeline_input_order);
      setState({ status: "ready", info });
    } catch (error) {
      setState({
        status: "error",
        message:
          error instanceof ApiError || error instanceof Error
            ? error.message
            : "Could not load the model info.",
      });
    }
  }, []);

  useEffect(() => {
    // This triggers an async loader; `load` calls `setState` inside its own
    // body (in a later microtask), not synchronously here. Standard "fetch on
    // mount" pattern, not the cascading-render case the rule targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-4 px-4 py-3">
          <div className="mr-auto min-w-0">
            <p className="truncate text-sm font-semibold text-slate-900 dark:text-slate-50">
              Insurance Claim Fraud Scoring
            </p>
            <p className="truncate text-xs text-slate-500 dark:text-slate-400">
              LightGBM + SHAP · explained per claim, with out-of-distribution warnings
            </p>
          </div>

          <nav className="flex items-center gap-1" aria-label="Main navigation">
            <NavLink current={route} target="score" navigate={navigate}>
              Score a claim
            </NavLink>
            <NavLink current={route} target="model-card" navigate={navigate}>
              Model card
            </NavLink>
          </nav>

          <ThemeToggle theme={theme} toggle={toggle} />
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {state.status === "loading" && <LoadingState showColdStartHint={showColdStartHint} />}
        {state.status === "error" && (
          // `load` swallows every error internally (try/catch) and never
          // rejects, so passing it here cannot produce an unhandled rejection.
          // eslint-disable-next-line @typescript-eslint/no-misused-promises
          <ErrorState message={state.message} onRetry={load} />
        )}
        {state.status === "ready" &&
          (route === "model-card" ? (
            <ModelCardPage info={state.info} />
          ) : (
            <ScorePage info={state.info} />
          ))}
      </main>

      <footer className="mx-auto max-w-7xl px-4 py-6 text-xs text-slate-400 dark:text-slate-500">
        <p>
          This is a demo. Model output should be used to prioritise human review,
          never to decline a claim on its own.
        </p>
        {state.status === "ready" && (
          <p className="mt-1">
            Model version {state.info.model_version} · API: {API_BASE_URL}
          </p>
        )}
      </footer>
    </div>
  );
}

function NavLink({
  current,
  target,
  navigate,
  children,
}: {
  current: string;
  target: "score" | "model-card";
  navigate: (route: "score" | "model-card") => void;
  children: React.ReactNode;
}) {
  const active = current === target;
  return (
    <RouteLink
      to={target}
      navigate={navigate}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
        active
          ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
          : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
      }`}
    >
      {children}
    </RouteLink>
  );
}

function LoadingState({ showColdStartHint }: { showColdStartHint: boolean }) {
  return (
    <div className="flex flex-col gap-4" aria-busy="true" aria-live="polite">
      <p className="text-sm text-slate-500 dark:text-slate-400">Loading model info…</p>
      {showColdStartHint && (
        <p className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          {COLD_START_HINT_MESSAGE}
        </p>
      )}
      <div className="h-32 animate-pulse rounded-xl bg-slate-200 dark:bg-slate-800" />
      <div className="h-64 animate-pulse rounded-xl bg-slate-200 dark:bg-slate-800" />
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div
      role="alert"
      className="rounded-xl border border-red-300 bg-red-50 p-5 dark:border-red-900 dark:bg-red-950/40"
    >
      <h2 className="text-sm font-semibold text-red-900 dark:text-red-200">
        Could not load the model info
      </h2>
      <p className="mt-2 text-sm leading-relaxed text-red-800 dark:text-red-300">{message}</p>
      <div className="mt-3 rounded-lg border border-red-200 bg-white/60 p-3 text-xs text-red-900 dark:border-red-900 dark:bg-slate-900/60 dark:text-red-200">
        <p className="font-medium">If you are running this locally:</p>
        <pre className="mt-1 overflow-x-auto font-mono text-[11px]">
          cd backend{"\n"}python -m uvicorn app.main:app --port 8000
        </pre>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="mt-3 rounded-lg bg-red-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-red-800"
      >
        Try again
      </button>
    </div>
  );
}
