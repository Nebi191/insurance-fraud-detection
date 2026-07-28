/**
 * Uygulama kabuğu: model bilgisini bir kez yükler, iki sayfayı yönlendirir.
 *
 * `/model-info` UYGULAMA AÇILIŞINDA ÇEKİLİR
 * -----------------------------------------
 * Form, seçenek listelerini ve aralıkları bu yanıttan kurduğu için o gelmeden
 * ekranda anlamlı bir şey gösteremeyiz. Beklerken iskelet, başarısız olursa
 * NEDENİ ile birlikte hata ekranı gösteriyoruz — "bir şeyler ters gitti"
 * demek, backend'i unutmuş bir geliştiriciye hiçbir şey söylemez.
 */

import { useCallback, useEffect, useState } from "react";

import { API_BASE_URL, ApiError, fetchModelInfo } from "./api";
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

  const load = useCallback(async () => {
    setState({ status: "loading" });
    try {
      const info = await fetchModelInfo();
      // Elle yazılmış alan listesi artefaktla uyuşmuyorsa SESSİZ kalmıyoruz:
      // eksik alan kullanıcının hiç göremediği bir girdi, fazla alan ise
      // `extra="forbid"` yüzünden her isteğin 422 alması demek.
      assertFieldsCoverContract(info.feature_list.pipeline_input_order);
      setState({ status: "ready", info });
    } catch (error) {
      setState({
        status: "error",
        message:
          error instanceof ApiError || error instanceof Error
            ? error.message
            : "Model bilgisi yüklenemedi.",
      });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-4 px-4 py-3">
          <div className="mr-auto min-w-0">
            <p className="truncate text-sm font-semibold text-slate-900 dark:text-slate-50">
              Sigorta Talebi Dolandırıcılık Skorlaması
            </p>
            <p className="truncate text-xs text-slate-500 dark:text-slate-400">
              LightGBM + SHAP · açıklamalı, dağılım dışı girdi uyarılı
            </p>
          </div>

          <nav className="flex items-center gap-1" aria-label="Ana gezinme">
            <NavLink current={route} target="score" navigate={navigate}>
              Skorlama
            </NavLink>
            <NavLink current={route} target="model-card" navigate={navigate}>
              Model kartı
            </NavLink>
          </nav>

          <ThemeToggle theme={theme} toggle={toggle} />
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {state.status === "loading" && <LoadingState />}
        {state.status === "error" && <ErrorState message={state.message} onRetry={load} />}
        {state.status === "ready" &&
          (route === "model-card" ? (
            <ModelCardPage info={state.info} />
          ) : (
            <ScorePage info={state.info} />
          ))}
      </main>

      <footer className="mx-auto max-w-7xl px-4 py-6 text-xs text-slate-400 dark:text-slate-500">
        <p>
          Bu bir demodur. Model kararları tek başına bir talebi reddetmek için
          değil, insan incelemesini önceliklendirmek için kullanılmalıdır.
        </p>
        {state.status === "ready" && (
          <p className="mt-1">
            Model sürümü {state.info.model_version} · API: {API_BASE_URL}
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

function LoadingState() {
  return (
    <div className="flex flex-col gap-4" aria-busy="true" aria-live="polite">
      <p className="text-sm text-slate-500 dark:text-slate-400">Model bilgisi yükleniyor…</p>
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
        Model bilgisi yüklenemedi
      </h2>
      <p className="mt-2 text-sm leading-relaxed text-red-800 dark:text-red-300">{message}</p>
      <div className="mt-3 rounded-lg border border-red-200 bg-white/60 p-3 text-xs text-red-900 dark:border-red-900 dark:bg-slate-900/60 dark:text-red-200">
        <p className="font-medium">Yerelde çalıştırıyorsanız:</p>
        <pre className="mt-1 overflow-x-auto font-mono text-[11px]">
          cd backend{"\n"}python -m uvicorn app.main:app --port 8000
        </pre>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="mt-3 rounded-lg bg-red-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-red-800"
      >
        Tekrar dene
      </button>
    </div>
  );
}
