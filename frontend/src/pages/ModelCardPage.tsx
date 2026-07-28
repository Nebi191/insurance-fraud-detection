/**
 * Model card sayfası — Faz 5.
 *
 * Bütün içerik `/model-info`'dan gelir; bu sayfada TEK BİR sabit metrik ya da
 * feature listesi yoktur. Sebep basit: model yeniden eğitilirse kopyalanmış bir
 * sayı sessizce yalan söylemeye başlar ve model card'ın tek işi doğru bilgi
 * vermektir.
 *
 * FAIRNESS BÖLÜMÜ SANSÜRLENMİYOR
 * ------------------------------
 * Model korunan nitelikleri (cinsiyet, yaş, medeni durum) feature olarak
 * kullanıyor ve bu sayfada açıkça yazıyor. Bir demo için bile saklamak yanlış
 * olurdu. Ama beyan ile ölçüm ayrı sütunlarda: "modele verildi" ile "eğitilmiş
 * ağaçlar gerçekten kullandı" aynı şey değil ve ikisini birden göstermek
 * okuyucuyu yanlış yönlendirmemenin tek yolu.
 */

import { fieldLabel } from "../fields";
import type { FairnessAttribute, ModelInfoResponse } from "../types";

export function ModelCardPage({ info }: { info: ModelInfoResponse }) {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50">Model kartı</h1>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
          {info.model_name} · sürüm {info.model_version} · {info.algorithm}
        </p>
        <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
          Eğitim tarihi: {formatDate(info.trained_at)}
        </p>
      </header>

      <Panel title="Başarım">
        <div className="grid gap-4 sm:grid-cols-3">
          <Metric
            label="Test PR-AUC"
            value={info.metrics.test_pr_auc.toFixed(4)}
            hint="Asıl ölçüt — görülmemiş veri"
          />
          <Metric
            label="Eğitim PR-AUC"
            value={info.metrics.train_pr_auc.toFixed(4)}
            hint="Aradaki fark ezberin göstergesi"
          />
          <Metric
            label="Ölçüt"
            value="PR-AUC"
            hint={info.metrics.metric_name}
          />
        </div>
        <p className="mt-4 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          Dengesiz sınıf probleminde ROC-AUC yanıltıcı iyimser okunur; bu yüzden
          PR-AUC raporlanıyor. Test setindeki pozitif oran{" "}
          {(info.dataset.positive_rate_test * 100).toFixed(1)}% — rastgele bir
          sınıflandırıcının PR-AUC'si de yaklaşık bu değerdir, karşılaştırma
          tabanı budur.
        </p>
      </Panel>

      <Panel title="Olasılık kalibrasyonu">
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm leading-relaxed text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          <p className="font-semibold">
            Kalibre edilmemiş ({info.probability_calibration.method}).
          </p>
          <p className="mt-1 text-xs">{info.probability_calibration.warning}</p>
        </div>
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
          Risk etiketi kuralı: <code className="font-mono">{info.risk_thresholds.rule}</code>
        </p>
      </Panel>

      <Panel title="Veri kümesi">
        <div className="grid gap-4 sm:grid-cols-4">
          <Metric label="Toplam satır" value={String(info.dataset.n_rows)} />
          <Metric label="Eğitim" value={String(info.dataset.n_train)} />
          <Metric label="Test" value={String(info.dataset.n_test)} />
          <Metric
            label="Pozitif oran"
            value={`${(info.dataset.positive_rate_train * 100).toFixed(1)}%`}
            hint="eğitim setinde"
          />
        </div>
        <p className="mt-4 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          Modele hiç girmeyen kolonlar (kimlik/PII ve ham tarihler):{" "}
          {info.feature_list.dropped_columns.map((column) => (
            <code
              key={column}
              className="mr-1 rounded bg-slate-100 px-1 py-0.5 font-mono text-[11px] dark:bg-slate-800"
            >
              {column}
            </code>
          ))}
        </p>
      </Panel>

      <Panel title="Feature etkisi">
        <p className="text-sm text-slate-600 dark:text-slate-300">
          Model {info.feature_influence.summary.n_features} feature ile eğitildi ama
          bunlardan yalnızca{" "}
          <strong>{info.feature_influence.summary.n_with_influence}</strong> tanesinde
          gerçekten dallanma yapıyor.{" "}
          <strong>{info.feature_influence.summary.n_without_influence}</strong> feature'ın
          split sayısı sıfır — bu kolonların değeri hiçbir tahmini etkilemiyor.
        </p>
        <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          {info.feature_influence.interpretation_note}
        </p>

        <h3 className="mt-4 mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
          Modelin kullanmadığı alanlar
        </h3>
        <ul className="flex flex-wrap gap-1.5">
          {info.feature_influence.summary.features_without_influence.map((name) => (
            <li
              key={name}
              className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400"
            >
              {fieldLabel(name)}
            </li>
          ))}
        </ul>

        <h3 className="mt-4 mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
          En çok kullanılan alanlar
        </h3>
        <TopFeatures info={info} />
      </Panel>

      <Panel title="Adillik (fairness)">
        <div className="rounded-lg border border-slate-300 bg-slate-50 p-3 text-xs leading-relaxed text-slate-700 dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-300">
          <p>
            <strong>Durum:</strong> {info.fairness.status} ·{" "}
            <strong>Denetim yapıldı mı:</strong>{" "}
            {info.fairness.audit_performed ? "evet" : "hayır"}
          </p>
          <p className="mt-2">{info.fairness.field_semantics}</p>
        </div>

        <FairnessTable
          title="Korunan nitelikler (model feature'ı olarak kullanılıyor)"
          rows={info.fairness.protected_attributes_used_as_features}
        />
        <FairnessTable
          title="Vekil (proxy) risk taşıyan nitelikler"
          rows={info.fairness.proxy_risk_attributes}
        />

        <h3 className="mt-5 mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
          Üretime almadan önce yapılması gerekenler
        </h3>
        <ul className="list-inside list-disc space-y-1.5 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
          {info.fairness.production_requirements.map((requirement) => (
            <li key={requirement}>{requirement}</li>
          ))}
        </ul>

        <p className="mt-4 border-t border-slate-200 pt-3 text-xs leading-relaxed whitespace-pre-line text-slate-500 dark:border-slate-800 dark:text-slate-400">
          {info.fairness.notes}
        </p>
      </Panel>

      <Panel title="Çalışma ortamı">
        <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
          Artefakt bu sürümlerle üretildi; yeniden üretim için gerekli.
        </p>
        <dl className="grid gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
          {Object.entries(info.library_versions).map(([name, version]) => (
            <div key={name} className="flex justify-between gap-2 border-b border-slate-100 py-1 dark:border-slate-800">
              <dt className="font-mono text-slate-600 dark:text-slate-300">{name}</dt>
              <dd className="font-mono tabular-nums text-slate-500 dark:text-slate-400">{version}</dd>
            </div>
          ))}
        </dl>
      </Panel>
    </div>
  );
}

function TopFeatures({ info }: { info: ModelInfoResponse }) {
  const top = Object.entries(info.feature_influence.features)
    .filter(([, entry]) => entry.has_influence)
    .sort((a, b) => b[1].split_count - a[1].split_count)
    .slice(0, 8);
  const max = top[0]?.[1].split_count ?? 1;

  return (
    <ul className="flex flex-col gap-1.5">
      {top.map(([name, entry]) => (
        <li key={name} className="flex items-center gap-3 text-xs">
          <span className="w-40 shrink-0 truncate text-slate-600 dark:text-slate-300">
            {fieldLabel(name)}
          </span>
          <span className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
            <span
              className="block h-full rounded-full bg-sky-600 dark:bg-sky-500"
              style={{ width: `${(entry.split_count / max) * 100}%` }}
            />
          </span>
          <span className="w-16 shrink-0 text-right tabular-nums text-slate-500 dark:text-slate-400">
            {entry.split_count} split
          </span>
        </li>
      ))}
    </ul>
  );
}

function FairnessTable({ title, rows }: { title: string; rows: FairnessAttribute[] }) {
  if (rows.length === 0) return null;

  return (
    <>
      <h3 className="mt-5 mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
        {title}
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[36rem] border-collapse text-left text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-slate-500 dark:border-slate-700 dark:text-slate-400">
              <th className="py-2 pr-3 font-medium">Alan</th>
              <th className="py-2 pr-3 font-medium">Dayanak</th>
              <th className="py-2 pr-3 font-medium">Modele verildi</th>
              <th className="py-2 pr-3 font-medium">Ölçülen etki</th>
              <th className="py-2 font-medium">Gerekçe</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.feature}
                className="border-b border-slate-100 align-top dark:border-slate-800"
              >
                <td className="py-2 pr-3 font-medium text-slate-700 dark:text-slate-200">
                  {fieldLabel(row.feature)}
                  {row.severity && (
                    <span className="ml-1.5 rounded bg-slate-100 px-1 py-0.5 text-[10px] text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                      {row.severity}
                    </span>
                  )}
                </td>
                <td className="py-2 pr-3 font-mono text-[11px] text-slate-500 dark:text-slate-400">
                  {row.basis}
                </td>
                <td className="py-2 pr-3">
                  <span className="font-medium text-slate-700 dark:text-slate-200">
                    {row.used_as_model_feature ? "evet" : "hayır"}
                  </span>
                </td>
                <td className="py-2 pr-3">
                  {row.has_influence ? (
                    <span className="font-medium text-amber-700 dark:text-amber-400">
                      var ({row.split_count} split)
                    </span>
                  ) : (
                    <span className="text-slate-500 dark:text-slate-400">
                      ölçülen etki yok (0 split)
                    </span>
                  )}
                </td>
                <td className="py-2 leading-relaxed text-slate-500 dark:text-slate-400">
                  {row.rationale}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900">
      <h2 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</h2>
      {children}
    </section>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <p className="text-xs text-slate-500 dark:text-slate-400">{label}</p>
      <p className="mt-0.5 text-xl font-semibold tabular-nums text-slate-900 dark:text-slate-50">
        {value}
      </p>
      {hint && <p className="text-[11px] text-slate-400 dark:text-slate-500">{hint}</p>}
    </div>
  );
}

function formatDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return new Intl.DateTimeFormat("tr-TR", { dateStyle: "long", timeStyle: "short" }).format(date);
}
