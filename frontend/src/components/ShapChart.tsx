/**
 * SHAP contribution chart — horizontal bars, deviation from zero.
 *
 * WHY NOT A PNG
 * -------------
 * The backend serves SHAP as raw JSON (`{feature, value, base_value}`), not as a
 * matplotlib image. The chart is drawn here, in the browser: the data stays
 * interactive (tooltips, sorting, filtering), it re-colours itself when the theme
 * changes, and the server never has to carry a plotting stack.
 *
 * WHAT IT SHOWS
 * -------------
 * Each bar is one feature's LOG-ODDS contribution. Bars extending right pushed
 * the risk score up, bars extending left pulled it down. They sum to the model's
 * raw margin:
 *
 *     sum(value) + base_value = raw margin,  sigmoid(raw margin) = fraud_probability
 *
 * The backend verifies that additivity at startup; the chart restates the same
 * identity underneath so the number does not look like it came out of nowhere.
 *
 * ZERO CONTRIBUTIONS ARE KEPT SEPARATE
 * ------------------------------------
 * The 16 features with `split_count = 0` contribute exactly 0.00. Showing them in
 * the main chart would mean 16 empty rows; they live in a separate collapsed list
 * instead, because "the model did not look at this" is information too.
 */

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fieldLabel } from "../fields";
import type { ShapValue } from "../types";

const POSITIVE = "var(--color-shap-positive)";
const NEGATIVE = "var(--color-shap-negative)";
const DEFAULT_VISIBLE = 10;

interface ChartRow {
  name: string;
  label: string;
  value: number;
}

export function ShapChart({ values }: { values: ShapValue[] }) {
  const [showAll, setShowAll] = useState(false);
  const [zeroOpen, setZeroOpen] = useState(false);

  const { contributing, zero, baseValue } = useMemo(() => {
    const rows: ChartRow[] = values.map((item) => ({
      name: item.feature,
      label: fieldLabel(item.feature),
      value: item.value,
    }));
    return {
      contributing: rows.filter((row) => row.value !== 0),
      zero: rows.filter((row) => row.value === 0),
      baseValue: values[0]?.base_value ?? 0,
    };
  }, [values]);

  const visible = showAll ? contributing : contributing.slice(0, DEFAULT_VISIBLE);
  // Recharts draws top to bottom; we do not reverse the rows because the backend
  // already sends them sorted by descending abs(value), so the strongest
  // contribution ends up on top.
  const chartHeight = Math.max(200, visible.length * 34 + 40);
  const total = contributing.reduce((sum, row) => sum + row.value, 0);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
          What produced this score?
        </h2>
        <span className="text-xs text-slate-500 dark:text-slate-400">
          {contributing.length} contributing, {zero.length} with zero contribution
        </span>
      </div>

      <p className="mb-3 text-xs text-slate-500 dark:text-slate-400">
        Bars show the <strong>log-odds</strong> contribution:{" "}
        <span className="font-medium text-[var(--color-shap-positive)]">right</span> pushed
        the risk up,{" "}
        <span className="font-medium text-[var(--color-shap-negative)]">left</span> pulled
        it down.
      </p>

      {visible.length === 0 ? (
        <p className="py-8 text-center text-sm text-slate-500 dark:text-slate-400">
          No feature produced a contribution.
        </p>
      ) : (
        <div style={{ height: chartHeight }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={visible}
              layout="vertical"
              margin={{ top: 4, right: 16, bottom: 4, left: 4 }}
            >
              <XAxis
                type="number"
                tick={{ fontSize: 11 }}
                stroke="currentColor"
                className="text-slate-400"
                tickFormatter={(value: number) => formatContribution(value)}
              />
              <YAxis
                type="category"
                dataKey="label"
                width={150}
                tick={{ fontSize: 11 }}
                stroke="currentColor"
                className="text-slate-500 dark:text-slate-400"
              />
              <ReferenceLine x={0} stroke="currentColor" className="text-slate-400" />
              <Tooltip
                cursor={{ fill: "currentColor", fillOpacity: 0.06 }}
                content={<ContributionTooltip />}
              />
              <Bar dataKey="value" radius={[3, 3, 3, 3]} isAnimationActive={false}>
                {visible.map((row) => (
                  <Cell key={row.name} fill={row.value >= 0 ? POSITIVE : NEGATIVE} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {contributing.length > DEFAULT_VISIBLE && (
        <button
          type="button"
          onClick={() => setShowAll((previous) => !previous)}
          className="mt-2 text-xs font-medium text-sky-700 underline underline-offset-2 hover:text-sky-900 dark:text-sky-400 dark:hover:text-sky-300"
        >
          {showAll
            ? `Show the top ${DEFAULT_VISIBLE} only`
            : `Show all ${contributing.length} contributing fields`}
        </button>
      )}

      <dl className="mt-4 grid gap-2 border-t border-slate-200 pt-3 text-xs text-slate-600 sm:grid-cols-3 dark:border-slate-800 dark:text-slate-400">
        <div>
          <dt className="font-medium text-slate-500 dark:text-slate-400">Base value</dt>
          <dd className="tabular-nums">{formatContribution(baseValue)}</dd>
        </div>
        <div>
          <dt className="font-medium text-slate-500 dark:text-slate-400">Sum of contributions</dt>
          <dd className="tabular-nums">{formatContribution(total)}</dd>
        </div>
        <div>
          <dt className="font-medium text-slate-500 dark:text-slate-400">Raw margin</dt>
          <dd className="tabular-nums">
            {formatContribution(baseValue + total)} → sigmoid →{" "}
            {(1 / (1 + Math.exp(-(baseValue + total))) * 100).toFixed(1)}%
          </dd>
        </div>
      </dl>

      {zero.length > 0 && (
        <div className="mt-3 border-t border-slate-200 pt-3 dark:border-slate-800">
          <button
            type="button"
            onClick={() => setZeroOpen((previous) => !previous)}
            aria-expanded={zeroOpen}
            className="text-xs font-medium text-slate-600 underline underline-offset-2 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200"
          >
            {zeroOpen ? "Hide" : "Show"} the {zero.length} fields contributing exactly zero
          </button>
          {zeroOpen && (
            <>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                The trained trees never branch on these columns; whatever their
                values, they cannot affect the score.
              </p>
              <ul className="mt-2 flex flex-wrap gap-1.5">
                {zero.map((row) => (
                  <li
                    key={row.name}
                    className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400"
                  >
                    {row.label}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </section>
  );
}

function ContributionTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: ChartRow }>;
}) {
  const row = payload?.[0]?.payload;
  if (!active || !row) return null;

  return (
    <div className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs shadow-lg dark:border-slate-600 dark:bg-slate-800">
      <p className="font-semibold text-slate-800 dark:text-slate-100">{row.label}</p>
      <p className="mt-0.5 tabular-nums text-slate-600 dark:text-slate-300">
        {formatContribution(row.value)} log-odds
      </p>
      <p className="mt-0.5 text-slate-500 dark:text-slate-400">
        {row.value >= 0 ? "Pushed the risk up" : "Pulled the risk down"}
      </p>
    </div>
  );
}

function formatContribution(value: number): string {
  const formatted = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
  return value > 0 ? `+${formatted}` : formatted;
}
