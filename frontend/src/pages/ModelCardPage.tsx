/**
 * Model card page — Phase 5.
 *
 * Every piece of content comes from `/model-info`; there is not a SINGLE
 * hard-coded metric or feature list on this page. The reason is simple: if the
 * model is retrained, a copied number starts lying silently, and a model card's
 * only job is to be accurate.
 *
 * THE FAIRNESS SECTION IS NOT CENSORED
 * ------------------------------------
 * The model uses protected attributes (sex, age, marital status) as features and
 * this page says so plainly. Hiding that would be wrong even in a demo. But the
 * declaration and the measurement live in separate columns: "handed to the
 * model" and "the trained trees actually used it" are not the same thing, and
 * showing both is the only way not to mislead the reader.
 */

import { fieldLabel } from "../fields";
import type { FairnessAttribute, ModelInfoResponse } from "../types";

export function ModelCardPage({ info }: { info: ModelInfoResponse }) {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50">Model card</h1>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
          {info.model_name} · version {info.model_version} · {info.algorithm}
        </p>
        <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
          Trained at: {formatDate(info.trained_at)}
        </p>
      </header>

      <Panel title="Performance">
        <div className="grid gap-4 sm:grid-cols-3">
          <Metric
            label="Test PR-AUC"
            value={info.metrics.test_pr_auc.toFixed(4)}
            hint="The metric that counts — unseen data"
          />
          <Metric
            label="Train PR-AUC"
            value={info.metrics.train_pr_auc.toFixed(4)}
            hint="The gap between the two indicates overfitting"
          />
          <Metric
            label="Metric"
            value="PR-AUC"
            hint={info.metrics.metric_name}
          />
        </div>
        <p className="mt-4 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          On an imbalanced problem ROC-AUC reads misleadingly optimistic, which is
          why PR-AUC is reported here. The positive rate in the test set is{" "}
          {(info.dataset.positive_rate_test * 100).toFixed(1)}% — a random
          classifier scores roughly that same PR-AUC, so that is the baseline to
          compare against.
        </p>
      </Panel>

      <Panel title="Probability calibration">
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm leading-relaxed text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          <p className="font-semibold">
            Not calibrated ({info.probability_calibration.method}).
          </p>
          <p className="mt-1 text-xs">{info.probability_calibration.warning}</p>
        </div>
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
          Risk label rule: <code className="font-mono">{info.risk_thresholds.rule}</code>
        </p>
      </Panel>

      <Panel title="Dataset">
        <div className="grid gap-4 sm:grid-cols-4">
          <Metric label="Total rows" value={String(info.dataset.n_rows)} />
          <Metric label="Train" value={String(info.dataset.n_train)} />
          <Metric label="Test" value={String(info.dataset.n_test)} />
          <Metric
            label="Positive rate"
            value={`${(info.dataset.positive_rate_train * 100).toFixed(1)}%`}
            hint="in the training set"
          />
        </div>
        <p className="mt-4 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          Columns that never reached the model (identifiers/PII and raw dates):{" "}
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

      <Panel title="Feature influence">
        <p className="text-sm text-slate-600 dark:text-slate-300">
          The model was trained on {info.feature_influence.summary.n_features} features
          but actually branches on only{" "}
          <strong>{info.feature_influence.summary.n_with_influence}</strong> of them.{" "}
          <strong>{info.feature_influence.summary.n_without_influence}</strong> features
          have a split count of zero — the values of those columns cannot affect any
          prediction.
        </p>
        <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          {info.feature_influence.interpretation_note}
        </p>

        <h3 className="mt-4 mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
          Fields the model does not use
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
          Most used fields
        </h3>
        <TopFeatures info={info} />
      </Panel>

      <Panel title="Fairness">
        <div className="rounded-lg border border-slate-300 bg-slate-50 p-3 text-xs leading-relaxed text-slate-700 dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-300">
          <p>
            <strong>Status:</strong> {info.fairness.status} ·{" "}
            <strong>Audit performed:</strong>{" "}
            {info.fairness.audit_performed ? "yes" : "no"}
          </p>
          <p className="mt-2">{info.fairness.field_semantics}</p>
        </div>

        <FairnessTable
          title="Protected attributes used as model features"
          rows={info.fairness.protected_attributes_used_as_features}
        />
        <FairnessTable
          title="Attributes carrying proxy risk"
          rows={info.fairness.proxy_risk_attributes}
        />

        <h3 className="mt-5 mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
          Required before production use
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

      <Panel title="Runtime environment">
        <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
          The artifact was produced with these versions; they are required to
          reproduce it.
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
            {entry.split_count} splits
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
              <th className="py-2 pr-3 font-medium">Field</th>
              <th className="py-2 pr-3 font-medium">Basis</th>
              <th className="py-2 pr-3 font-medium">Given to model</th>
              <th className="py-2 pr-3 font-medium">Measured influence</th>
              <th className="py-2 font-medium">Rationale</th>
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
                    {row.used_as_model_feature ? "yes" : "no"}
                  </span>
                </td>
                <td className="py-2 pr-3">
                  {row.has_influence ? (
                    <span className="font-medium text-amber-700 dark:text-amber-400">
                      yes ({row.split_count} splits)
                    </span>
                  ) : (
                    <span className="text-slate-500 dark:text-slate-400">
                      none measured (0 splits)
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
  return new Intl.DateTimeFormat("en-US", { dateStyle: "long", timeStyle: "short" }).format(date);
}
