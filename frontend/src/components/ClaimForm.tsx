/**
 * Talep formu — 34 alan, hepsi opsiyonel.
 *
 * DÜZEN KARARI
 * ------------
 * Skoru fiilen sürükleyen altı alan AÇIK bir grupta, kalanlar katlanmış
 * gruplarda. Hiçbir alan gizlenmiyor — grup başlıkları kaç alan içerdiğini ve
 * kaçının modelce kullanılmadığını da yazıyor, çünkü "Olay (11 alan, 9'u
 * kullanılmıyor)" başlığı tek başına bir bulgu.
 *
 * BOŞ ALAN = GÖNDERİLMEZ
 * `buildPayload` yalnızca doldurulmuş alanları paketler. Backend kalanları
 * eğitim medyanı/moduyla doldurur ve guardrail SADECE gönderilen alanları
 * kontrol eder — boş bir formun 34 uyarı basmamasının sebebi bu.
 */

import { useState } from "react";

import { ALL_GROUPS, HIGHLIGHT_GROUP, type FieldGroup } from "../fields";
import type { ModelInfoResponse, PredictRequest } from "../types";
import { FieldControl } from "./FieldControl";

export type FormValues = Record<string, string>;

export function buildPayload(values: FormValues, info: ModelInfoResponse): PredictRequest {
  const payload: PredictRequest = {};
  for (const [name, raw] of Object.entries(values)) {
    const value = raw.trim();
    if (value === "") continue; // boş = gönderme, varsayılan kullanılsın

    if (info.training_ranges[name]?.type === "numeric") {
      const parsed = Number(value);
      // NaN'ı göndermiyoruz: backend 422 döndürürdü ve kullanıcı hatayı
      // anlamsız bir sunucu mesajından öğrenirdi. Sayı kutusu zaten harf
      // kabul etmiyor; bu ikinci bir emniyet.
      if (Number.isFinite(parsed)) payload[name] = parsed;
    } else {
      payload[name] = value;
    }
  }
  return payload;
}

interface ClaimFormProps {
  info: ModelInfoResponse;
  values: FormValues;
  onChange: (name: string, value: string) => void;
  onSubmit: () => void;
  onReset: () => void;
  fieldErrors: Record<string, string>;
  oodFields: Set<string>;
  submitting: boolean;
}

export function ClaimForm({
  info,
  values,
  onChange,
  onSubmit,
  onReset,
  fieldErrors,
  oodFields,
  submitting,
}: ClaimFormProps) {
  const filledCount = Object.values(values).filter((value) => value.trim() !== "").length;

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="flex flex-col gap-4"
    >
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300">
        <p>
          <strong className="font-semibold text-slate-800 dark:text-slate-100">
            Bütün alanlar isteğe bağlı.
          </strong>{" "}
          Boş bıraktıklarınız eğitim verisinin medyan/mod değerleriyle doldurulur —
          hiçbir şey doldurmadan da skor alabilirsiniz.
        </p>
        <p className="mt-2">
          <span className="font-medium">{info.feature_influence.summary.n_without_influence}</span>{" "}
          alan <em>model kullanmıyor</em> rozetiyle işaretli: bu eğitilmiş modelin
          ağaçları o kolonlarda hiç dallanmıyor, dolayısıyla değerleri skoru
          etkilemiyor.
        </p>
      </div>

      {ALL_GROUPS.map((group) => (
        <FieldGroupSection
          key={group.id}
          group={group}
          info={info}
          values={values}
          onChange={onChange}
          fieldErrors={fieldErrors}
          oodFields={oodFields}
          defaultOpen={group.id === HIGHLIGHT_GROUP.id}
        />
      ))}

      <div className="sticky bottom-0 -mx-1 flex flex-wrap items-center gap-3 border-t border-slate-200 bg-slate-50/95 px-1 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-950/95">
        <button
          type="submit"
          disabled={submitting}
          className="rounded-lg bg-sky-700 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-sky-600 dark:hover:bg-sky-500"
        >
          {submitting ? "Skorlanıyor…" : "Skorla"}
        </button>
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
        >
          Formu temizle
        </button>
        <span className="text-xs text-slate-500 dark:text-slate-400">
          {filledCount === 0
            ? "Hiçbir alan doldurulmadı — hepsi varsayılanla gidecek."
            : `${filledCount} alan dolduruldu, kalan ${34 - filledCount} alan varsayılanla gidecek.`}
        </span>
      </div>
    </form>
  );
}

function FieldGroupSection({
  group,
  info,
  values,
  onChange,
  fieldErrors,
  oodFields,
  defaultOpen,
}: {
  group: FieldGroup;
  info: ModelInfoResponse;
  values: FormValues;
  onChange: (name: string, value: string) => void;
  fieldErrors: Record<string, string>;
  oodFields: Set<string>;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  const deadCount = group.fields.filter(
    (field) => info.feature_influence.features[field.name]?.has_influence === false,
  ).length;
  const errorCount = group.fields.filter((field) => fieldErrors[field.name]).length;
  const oodCount = group.fields.filter((field) => oodFields.has(field.name)).length;
  const sectionId = `group-${group.id}`;

  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <h2>
        <button
          type="button"
          onClick={() => setOpen((previous) => !previous)}
          aria-expanded={open}
          aria-controls={sectionId}
          className="flex w-full items-center gap-3 px-4 py-3 text-left transition hover:bg-slate-50 dark:hover:bg-slate-800/60"
        >
          <span
            aria-hidden="true"
            className={`text-slate-400 transition-transform ${open ? "rotate-90" : ""}`}
          >
            ▶
          </span>
          <span className="flex-1">
            <span className="block text-sm font-semibold text-slate-800 dark:text-slate-100">
              {group.title}
            </span>
            <span className="block text-xs text-slate-500 dark:text-slate-400">
              {group.fields.length} alan
              {deadCount > 0 && `, ${deadCount}'i model tarafından kullanılmıyor`}
            </span>
          </span>
          {errorCount > 0 && (
            <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-semibold text-red-700 dark:bg-red-950 dark:text-red-300">
              {errorCount} hata
            </span>
          )}
          {oodCount > 0 && (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800 dark:bg-amber-950 dark:text-amber-300">
              {oodCount} dağılım dışı
            </span>
          )}
        </button>
      </h2>

      {open && (
        <div id={sectionId} className="border-t border-slate-200 px-4 py-4 dark:border-slate-800">
          <p className="mb-4 text-xs text-slate-500 dark:text-slate-400">{group.description}</p>
          <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2">
            {group.fields.map((field) => {
              const range = info.training_ranges[field.name];
              const fallback = info.defaults[field.name];
              const influence = info.feature_influence.features[field.name];
              // Sözleşme uyumu açılışta doğrulanıyor; yine de tip düzeyinde
              // güvenli davranıp eksik alanı sessizce atlamıyoruz.
              if (!range || !fallback || !influence) return null;

              return (
                <FieldControl
                  key={field.name}
                  meta={field}
                  range={range}
                  fallback={fallback}
                  splitCount={influence.split_count}
                  hasInfluence={influence.has_influence}
                  value={values[field.name] ?? ""}
                  onChange={(value) => onChange(field.name, value)}
                  error={fieldErrors[field.name]}
                  outOfDistribution={oodFields.has(field.name)}
                />
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
