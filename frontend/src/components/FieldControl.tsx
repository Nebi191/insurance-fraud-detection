/**
 * Tek bir form alanı.
 *
 * Girdi tipi ELLE değil `/model-info -> training_ranges` üzerinden belirlenir:
 * `categorical` ise açılır liste (seçenekler eğitimde görülen kategoriler),
 * `numeric` ise sayı kutusu. Böylece artefakt değişince form kendini düzeltir.
 *
 * Alan boş bırakılırsa istekte HİÇ GÖNDERİLMEZ; backend eğitim medyanı/moduyla
 * doldurur. Bu yüzden placeholder olarak varsayılan değeri gösteriyoruz:
 * kullanıcı "boş bıraktığım alan neyle dolduruldu?" sorusunun cevabını
 * tahmin etmek zorunda kalmıyor.
 */

import type { DefaultInfo, TrainingRange } from "../types";
import { valueLabel, type FieldMeta } from "../fields";
import { InfluenceBadge } from "./InfluenceBadge";

export interface FieldControlProps {
  meta: FieldMeta;
  range: TrainingRange;
  fallback: DefaultInfo;
  splitCount: number;
  hasInfluence: boolean;
  value: string;
  onChange: (value: string) => void;
  /** Backend'den gelen alan bazlı 422 mesajı. */
  error?: string | undefined;
  /** Bu alan son tahminde dağılım dışı olarak işaretlendi mi? */
  outOfDistribution?: boolean;
}

export function FieldControl({
  meta,
  range,
  fallback,
  splitCount,
  hasInfluence,
  value,
  onChange,
  error,
  outOfDistribution,
}: FieldControlProps) {
  const inputId = `field-${meta.name}`;
  const describedBy: string[] = [];
  if (meta.hint) describedBy.push(`${inputId}-hint`);
  if (error) describedBy.push(`${inputId}-error`);
  if (outOfDistribution) describedBy.push(`${inputId}-ood`);

  const baseInputClass = [
    "w-full rounded-lg border px-3 py-2 text-sm transition",
    "bg-white text-slate-900 placeholder:text-slate-400",
    "dark:bg-slate-900 dark:text-slate-100 dark:placeholder:text-slate-500",
    error
      ? "border-red-500 dark:border-red-500"
      : outOfDistribution
        ? "border-amber-500 dark:border-amber-500"
        : "border-slate-300 dark:border-slate-700",
  ].join(" ");

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor={inputId} className="text-sm font-medium text-slate-700 dark:text-slate-200">
          {meta.label}
        </label>
        {!hasInfluence && <InfluenceBadge splitCount={splitCount} />}
      </div>

      {meta.hint && (
        <p id={`${inputId}-hint`} className="text-xs text-slate-500 dark:text-slate-400">
          {meta.hint}
        </p>
      )}

      {range.type === "categorical" ? (
        <select
          id={inputId}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className={baseInputClass}
          aria-describedby={describedBy.join(" ") || undefined}
          aria-invalid={error ? true : undefined}
        >
          {/* Boş seçenek = "gönderme, varsayılanı kullan". */}
          <option value="">— varsayılan: {valueLabel(String(fallback.value))} —</option>
          {(range.categories ?? []).map((option) => (
            <option key={option} value={option}>
              {valueLabel(option)}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={inputId}
          type="number"
          inputMode="decimal"
          step={meta.step ?? 1}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={`varsayılan: ${fallback.value}`}
          className={baseInputClass}
          aria-describedby={describedBy.join(" ") || undefined}
          aria-invalid={error ? true : undefined}
        />
      )}

      {range.type === "numeric" && range.min != null && range.max != null && (
        <p className="text-xs text-slate-400 dark:text-slate-500">
          Eğitimde görülen aralık: {formatNumber(range.min)} – {formatNumber(range.max)}
        </p>
      )}

      {outOfDistribution && (
        <p
          id={`${inputId}-ood`}
          className="flex items-start gap-1.5 text-xs font-medium text-amber-700 dark:text-amber-500"
        >
          <span aria-hidden="true">▲</span>
          <span>
            Model bu değeri eğitimde görmedi. Skor yine hesaplandı ama bu alan için
            temkinli okunmalı.
          </span>
        </p>
      )}

      {error && (
        <p id={`${inputId}-error`} role="alert" className="text-xs font-medium text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 2 }).format(value);
}
