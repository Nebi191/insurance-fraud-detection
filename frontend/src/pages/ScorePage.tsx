import { Suspense, lazy, useCallback, useMemo, useState } from "react";

import { ApiError, predict } from "../api";
import { ClaimForm, buildPayload, type FormValues } from "../components/ClaimForm";
import type { ModelInfoResponse, PredictResponse } from "../types";

/**
 * Sonuç kartı TEMBEL yükleniyor çünkü içindeki `ShapChart` recharts'a bağlı ve
 * recharts tek başına paketin yarısından fazlası. Kullanıcı "Skorla"ya basana
 * kadar grafik kütüphanesini indirmesinin bir anlamı yok: ilk açılış belirgin
 * biçimde hızlanıyor, sonuç gelirken küçük bir gecikme oluşuyor (o sırada
 * zaten ağdan yanıt bekleniyor).
 */
const ResultCard = lazy(() =>
  import("../components/ResultCard").then((module) => ({ default: module.ResultCard })),
);

/**
 * Skorlama sayfası: solda form, sağda sonuç.
 *
 * SONUÇ ESKİMEZ, TEMİZLENİR
 * -------------------------
 * Kullanıcı formu değiştirdiğinde önceki sonucu ekranda BIRAKMIYORUZ ama
 * silmiyoruz da — "bayat" olarak işaretliyoruz. Sessizce durması, değiştirilen
 * girdiye ait sanılmasına yol açardı; tamamen kaldırmak ise karşılaştırma
 * imkânını yok ederdi.
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
    // Kullanıcı alana dokununca o alanın hatası kalkar; hâlâ geçersizse
    // sunucu bir sonraki gönderimde tekrar söyler.
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
            ? "Bazı alanlar kabul edilmedi — ayrıntı alanların altında."
            : "Girdi doğrulamadan geçmedi.",
        );
      } else {
        setFormError(
          error instanceof ApiError ? error.message : "Beklenmeyen bir hata oluştu.",
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
                Form değişti — aşağıdaki sonuç önceki girdilere ait. Yeniden
                skorlayın.
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
        Sonuç burada görünecek
      </p>
      <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Formu doldurup <strong>Skorla</strong>'ya basın. Hiçbir alanı doldurmadan
        da deneyebilirsiniz — model eksik alanları eğitim verisinin medyan/mod
        değerleriyle tamamlar.
      </p>
    </section>
  );
}
