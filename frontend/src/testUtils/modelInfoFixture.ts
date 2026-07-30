/**
 * A minimal-but-contract-shaped `/model-info` response for tests.
 *
 * It is built from `ALL_FIELD_NAMES` (the same list `ClaimForm` renders from)
 * rather than hand-listing all 34 fields, so it can never silently go stale
 * against `src/fields.ts`. Every field is declared numeric here purely to keep
 * the fixture small — these tests exercise `ScorePage`'s request-orchestration
 * logic, not the real categorical/numeric split, which is already covered by
 * a live `/model-info` response.
 */
import { ALL_FIELD_NAMES } from "../fields";
import type { ModelInfoResponse, PredictResponse } from "../types";

export function buildModelInfoFixture(): ModelInfoResponse {
  const trainingRanges: ModelInfoResponse["training_ranges"] = {};
  const physicalRanges: ModelInfoResponse["physical_ranges"] = {};
  const defaults: ModelInfoResponse["defaults"] = {};
  const features: ModelInfoResponse["feature_influence"]["features"] = {};

  for (const name of ALL_FIELD_NAMES) {
    trainingRanges[name] = { type: "numeric", dtype: "float64", min: 0, max: 100 };
    // Deliberately wider than `trainingRanges` above, mirroring the real
    // `witnesses` case (trained on 0-3, API accepts 0-10) — a fixture where
    // the two ranges were identical could not tell "both shown" apart from
    // "one accidentally duplicated".
    physicalRanges[name] = { min: -100, max: 1000 };
    defaults[name] = { value: 0, dtype: "float64", strategy: "median" };
    features[name] = { split_count: 1, gain: 1, has_influence: true };
  }

  return {
    model_name: "insurance-fraud-detection-test",
    model_version: "test-0.0.0",
    algorithm: "LightGBM (LGBMClassifier)",
    trained_at: "2026-01-01T00:00:00+00:00",
    metrics: { train_pr_auc: 0.8, test_pr_auc: 0.65, metric_name: "average_precision_score" },
    dataset: {
      n_rows: 1000,
      n_train: 800,
      n_test: 200,
      test_size: 0.2,
      random_state: 42,
      positive_rate_train: 0.25,
      positive_rate_test: 0.25,
    },
    feature_list: {
      pipeline_input_order: ALL_FIELD_NAMES,
      categorical_features: [],
      numeric_features: ALL_FIELD_NAMES,
      dropped_columns: [],
      target: "fraud_reported",
    },
    feature_influence: {
      description: "test fixture",
      measured_from: "test fixture",
      interpretation_note: "test fixture",
      summary: {
        n_features: ALL_FIELD_NAMES.length,
        n_with_influence: ALL_FIELD_NAMES.length,
        n_without_influence: 0,
        features_without_influence: [],
      },
      features,
    },
    training_ranges: trainingRanges,
    physical_ranges: physicalRanges,
    defaults,
    fairness: {
      status: "test fixture",
      field_semantics: "test fixture",
      protected_attributes_used_as_features: [],
      proxy_risk_attributes: [],
      audit_performed: false,
      audit_metrics_computed: [],
      intended_use: "test fixture",
      production_requirements: [],
      notes: "test fixture",
    },
    library_versions: {},
    risk_thresholds: { low_below: 0.35, high_at_or_above: 0.65, rule: "test fixture" },
    probability_calibration: {
      calibrated: false,
      method: "test fixture",
      warning: "test fixture",
    },
  };
}

export function buildPredictResponseFixture(
  overrides: Partial<PredictResponse> = {},
): PredictResponse {
  return {
    fraud_probability: 0.42,
    risk_level: "medium",
    shap_values: [],
    out_of_distribution_warnings: [],
    ...overrides,
  };
}
