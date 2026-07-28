/**
 * The "not used by the model" badge.
 *
 * WHY IT EXISTS
 * -------------
 * 16 of this model's 34 features have a split count of ZERO in the booster: the
 * trees never branch on those columns, so their SHAP contributions are exactly
 * 0.0 and changing the value does not move the score by a single bit (proven on
 * the backend across 40+ value combinations by
 * `test_dead_features_change_neither_the_score_nor_their_own_shap_value`).
 *
 * HIDING those fields would have made the model's limitation invisible. The
 * badge turns a weakness into transparency: "my API also tells you which inputs
 * it is pointless to ask for."
 *
 * DATA SOURCE: `/model-info -> feature_influence`. The list is NOT hard-coded;
 * retraining the model updates the badges by itself.
 */

export function InfluenceBadge({ splitCount }: { splitCount: number }) {
  return (
    <span
      className="group/badge relative inline-flex shrink-0 items-center gap-1 rounded-full border border-slate-300 bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400"
      // Focusable so the explanation is reachable without a mouse.
      tabIndex={0}
    >
      <svg viewBox="0 0 24 24" className="size-3" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <path d="M6 6l12 12" strokeLinecap="round" />
      </svg>
      unused by model
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-0 z-20 mb-1.5 hidden w-64 rounded-lg border border-slate-300 bg-white p-2.5 text-[11px] leading-relaxed font-normal text-slate-700 shadow-lg group-hover/badge:block group-focus-within/badge:block dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
      >
        The trained trees never branch on this column ({splitCount} splits).
        Changing its value has <strong>no effect on the score</strong>, and its
        SHAP contribution is always exactly 0.00.
        <span className="mt-1.5 block text-slate-500 dark:text-slate-400">
          This does not mean the field is unimportant — only that <em>this</em>{" "}
          trained model does not look at it.
        </span>
      </span>
    </span>
  );
}
