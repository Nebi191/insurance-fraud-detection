/**
 * Skor kartı + dağılım dışı (OOD) uyarı banner'ı.
 *
 * OLASILIK "YÜZDE İHTİMAL" DEĞİLDİR
 * ---------------------------------
 * Model `scale_pos_weight ≈ 3.04` ile eğitildi: azınlık sınıfı üç kat
 * ağırlıklandırıldı ve bu `predict_proba` çıktısını sistematik olarak yukarı
 * şişiriyor. Yani 0,73 "bu talebin %73 ihtimalle dolandırıcılık olduğu"
 * anlamına GELMEZ; yalnızca "diğer taleplere göre yüksek sırada" demektir.
 *
 * Bu uyarı kartın üzerinde duruyor, dipnotta değil — bir sigorta eksperi sayıyı
 * yanlış okursa demonun tamamı yanlış anlaşılır. Metin backend'den geliyor
 * (`/model-info -> probability_calibration.warning`), frontend kendi kopyasını
 * uydurmuyor.
 */

import { fieldLabel } from "../fields";
import type { ModelInfoResponse, PredictResponse, RiskLevel } from "../types";
import { ShapChart } from "./ShapChart";

const RISK_PRESENTATION: Record<RiskLevel, { label: string; className: string; bar: string }> = {
  low: {
    label: "Düşük risk",
    className:
      "border-[var(--color-risk-low)] bg-[color-mix(in_oklch,var(--color-risk-low)_12%,transparent)] text-[var(--color-risk-low)]",
    bar: "bg-[var(--color-risk-low)]",
  },
  medium: {
    label: "Orta risk",
    className:
      "border-[var(--color-risk-medium)] bg-[color-mix(in_oklch,var(--color-risk-medium)_14%,transparent)] text-[var(--color-risk-medium)]",
    bar: "bg-[var(--color-risk-medium)]",
  },
  high: {
    label: "Yüksek risk",
    className:
      "border-[var(--color-risk-high)] bg-[color-mix(in_oklch,var(--color-risk-high)_12%,transparent)] text-[var(--color-risk-high)]",
    bar: "bg-[var(--color-risk-high)]",
  },
};

export function ResultCard({
  result,
  info,
}: {
  result: PredictResponse;
  info: ModelInfoResponse;
}) {
  const presentation = RISK_PRESENTATION[result.risk_level];
  const percentage = result.fraud_probability * 100;

  return (
    <div className="flex flex-col gap-4">
      {result.out_of_distribution_warnings.length > 0 && (
        <OodBanner fields={result.out_of_distribution_warnings} info={info} />
      )}

      <section className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-medium tracking-wide text-slate-500 uppercase dark:text-slate-400">
              Dolandırıcılık skoru
            </p>
            <p className="mt-1 text-4xl font-bold tabular-nums text-slate-900 dark:text-slate-50">
              {percentage.toFixed(1)}
              <span className="text-2xl text-slate-400">%</span>
            </p>
          </div>
          <span
            className={`rounded-full border px-3 py-1 text-sm font-semibold ${presentation.className}`}
          >
            {presentation.label}
          </span>
        </div>

        {/* Eşik çizgileri skalanın neresinde olduğumuzu gösterir; eşikler
            backend'den gelir, frontend kendi kopyasını tutmaz. */}
        <div className="relative mt-4 h-2.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
          <div
            className={`h-full rounded-full transition-all ${presentation.bar}`}
            style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
          />
        </div>
        <div className="mt-1.5 flex justify-between text-[11px] text-slate-400 dark:text-slate-500">
          <span>0</span>
          <span>orta eşiği: {(info.risk_thresholds.low_below * 100).toFixed(0)}%</span>
          <span>yüksek eşiği: {(info.risk_thresholds.high_at_or_above * 100).toFixed(0)}%</span>
          <span>100</span>
        </div>

        <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs leading-relaxed text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          <p className="font-semibold">Bu yüzde bir olasılık değil, bir sıralama skorudur.</p>
          <p className="mt-1">{info.probability_calibration.warning}</p>
        </div>
      </section>

      <ShapChart values={result.shap_values} />
    </div>
  );
}

/**
 * Dağılım dışı girdi uyarısı.
 *
 * Guardrail'in ürettiği liste SIRALI gelir: modelin gerçekten kullandığı alanlar
 * başta. Banner bu sırayı koruyor ve ölü alanları ayrıca işaretliyor — çünkü
 * `umbrella_limit` için "aralık dışı" uyarısı ile `age` için aynı uyarı aynı
 * ağırlıkta değil: ilkinde skor gerçekten etkilenmemiştir.
 */
function OodBanner({ fields, info }: { fields: string[]; info: ModelInfoResponse }) {
  return (
    <section
      role="status"
      className="rounded-xl border border-amber-400 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/40"
    >
      <div className="flex items-start gap-3">
        <span aria-hidden="true" className="mt-0.5 text-lg leading-none text-amber-600 dark:text-amber-500">
          ▲
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-amber-900 dark:text-amber-200">
            {fields.length} alan modelin eğitimde gördüğü aralığın dışında
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-amber-800 dark:text-amber-300">
            Skor yine hesaplandı — uyarı tahmini engellemez. Ama model bu değerleri
            hiç görmedi, dolayısıyla bu talep için verdiği sayı sessizce
            ekstrapolasyondur. Ağaç modelleri aralık dışına <em>uzanmaz</em>: gördüğü
            en uç değere kadar öğrendiğini uygular, ötesini ayırt edemez.
          </p>

          <ul className="mt-3 flex flex-wrap gap-2">
            {fields.map((name) => {
              const range = info.training_ranges[name];
              const dead = info.feature_influence.features[name]?.has_influence === false;
              return (
                <li
                  key={name}
                  className="rounded-lg border border-amber-300 bg-white px-2.5 py-1 text-xs dark:border-amber-800 dark:bg-slate-900"
                >
                  <span className="font-medium text-amber-900 dark:text-amber-200">
                    {fieldLabel(name)}
                  </span>
                  {range?.min != null && range?.max != null && (
                    <span className="ml-1.5 tabular-nums text-amber-700 dark:text-amber-400">
                      (eğitim: {formatCompact(range.min)}–{formatCompact(range.max)})
                    </span>
                  )}
                  {dead && (
                    <span className="ml-1.5 text-slate-500 dark:text-slate-400">
                      · model bu alanı kullanmıyor, skor etkilenmedi
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </section>
  );
}

function formatCompact(value: number): string {
  return new Intl.NumberFormat("tr-TR", {
    notation: Math.abs(value) >= 100_000 ? "compact" : "standard",
    maximumFractionDigits: 2,
  }).format(value);
}
