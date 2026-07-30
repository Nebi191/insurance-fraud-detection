/**
 * TypeScript mirror of the backend API contract.
 *
 * Source: `backend/app/schemas.py` (and the published `/openapi.json`).
 * These types are hand-written but not invented — each one corresponds directly
 * to a model in the schema. The validation in `src/api.ts` also checks at RUNTIME
 * that a response really has this shape; a type assertion alone guarantees
 * nothing.
 */

/** `POST /predict` body. All 34 fields are optional. */
export type PredictRequest = Record<string, string | number | null>;

/**
 * One feature's SHAP contribution.
 *
 * `value` and `base_value` live in LOG-ODDS space, not probability:
 *   sum(value) + base_value = raw margin,  sigmoid(raw margin) = fraudProbability
 */
export interface ShapValue {
  feature: string;
  value: number;
  base_value: number;
}

export type RiskLevel = "low" | "medium" | "high";

export interface PredictResponse {
  fraud_probability: number;
  risk_level: RiskLevel;
  /** All 34 features, sorted by descending abs(value). */
  shap_values: ShapValue[];
  /**
   * Names of NUMERIC fields that fall outside the training range.
   *
   * Only fields actually SENT in the request are checked; categorical fields
   * never appear here (an invalid category produces a 422, not a warning). The
   * ordering is meaningful: fields the model actually uses come first.
   */
  out_of_distribution_warnings: string[];
}

/* -------------------------------------------------------------------------- */
/* GET /model-info                                                            */
/* -------------------------------------------------------------------------- */

export interface TrainingRange {
  type: "numeric" | "categorical";
  dtype: string;
  min?: number | null;
  max?: number | null;
  categories?: string[] | null;
}

/**
 * The API's hard (Pydantic `ge`/`le`) bound for one NUMERIC field — distinct
 * from `TrainingRange`, which is the min/max actually observed in the
 * training data. A value inside `PhysicalRange` but outside `TrainingRange`
 * is accepted (200) with an out-of-distribution warning; a value outside
 * `PhysicalRange` is rejected (422) before it ever reaches the model.
 */
export interface PhysicalRange {
  min: number;
  max: number;
}

export interface DefaultInfo {
  value: number | string;
  dtype: string;
  strategy: "median" | "mode";
}

export interface FeatureInfluenceEntry {
  split_count: number;
  gain: number;
  /** `split_count > 0` — did the model actually use this feature? */
  has_influence: boolean;
}

export interface FeatureInfluence {
  description: string;
  measured_from: string;
  interpretation_note: string;
  summary: {
    n_features: number;
    n_with_influence: number;
    n_without_influence: number;
    features_without_influence: string[];
  };
  features: Record<string, FeatureInfluenceEntry>;
}

export interface FairnessAttribute {
  feature: string;
  basis: string;
  rationale: string;
  /** DECLARATION: the feature was handed to the model as an input. */
  used_as_model_feature: boolean;
  /** MEASUREMENT: did the trained trees actually split on it? */
  split_count: number;
  has_influence: boolean;
  severity?: string | null;
}

export interface FairnessInfo {
  status: string;
  field_semantics: string;
  protected_attributes_used_as_features: FairnessAttribute[];
  proxy_risk_attributes: FairnessAttribute[];
  audit_performed: boolean;
  audit_metrics_computed: string[];
  intended_use: string;
  production_requirements: string[];
  notes: string;
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  algorithm: string;
  trained_at: string;
  metrics: {
    train_pr_auc: number;
    test_pr_auc: number;
    metric_name: string;
  };
  dataset: {
    n_rows: number;
    n_train: number;
    n_test: number;
    test_size: number;
    random_state: number;
    positive_rate_train: number;
    positive_rate_test: number;
  };
  feature_list: {
    pipeline_input_order: string[];
    categorical_features: string[];
    numeric_features: string[];
    /** PII column NAMES — not a leak, but a statement that they never reached the model. */
    dropped_columns: string[];
    target: string;
  };
  feature_influence: FeatureInfluence;
  training_ranges: Record<string, TrainingRange>;
  /** Numeric fields only — a categorical field has no entry here. */
  physical_ranges: Record<string, PhysicalRange>;
  defaults: Record<string, DefaultInfo>;
  fairness: FairnessInfo;
  library_versions: Record<string, string>;
  risk_thresholds: {
    low_below: number;
    high_at_or_above: number;
    rule: string;
  };
  probability_calibration: {
    calibrated: boolean;
    method: string;
    warning: string;
  };
}

export interface HealthResponse {
  status: "ok";
  model_loaded: boolean;
  model_version: string;
}

/* -------------------------------------------------------------------------- */
/* Errors                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * The backend's 422 body.
 *
 * The `input` and `ctx` fields are absent DELIBERATELY: the backend does not
 * echo the submitted raw value back on a bad request, so claim data cannot leak
 * into logs or a console. All we get is which field (`loc`) and what is wrong
 * (`msg`).
 */
export interface ValidationErrorItem {
  loc: string[];
  msg: string;
  type: string;
}

export interface ValidationErrorBody {
  detail: ValidationErrorItem[];
}
