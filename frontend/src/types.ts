/**
 * Backend API sözleşmesinin TypeScript karşılığı.
 *
 * Kaynak: `backend/app/schemas.py` (ve yayınlanan `/openapi.json`).
 * Buradaki tipler ELLE yazıldı ama uydurulmadı — her biri şemadaki bir modele
 * birebir karşılık gelir. `src/api.ts` içindeki doğrulama, yanıtın gerçekten bu
 * şekle uyduğunu çalışma zamanında da kontrol eder; tip iddiası tek başına
 * güvence değildir.
 */

/** `POST /predict` gövdesi. 34 alanın hepsi opsiyonel. */
export type PredictRequest = Record<string, string | number | null>;

/**
 * Tek bir feature'ın SHAP katkısı.
 *
 * `value` ve `baseValue` LOG-ODDS uzayındadır, olasılık değil:
 *   sum(value) + base_value = ham skor,  sigmoid(ham skor) = fraudProbability
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
  /** 34 feature, abs(value) azalan sırada. */
  shap_values: ShapValue[];
  /**
   * Eğitim aralığı dışında kalan SAYISAL alan adları.
   *
   * Yalnızca istekte GÖNDERİLEN alanlar kontrol edilir; kategorik alanlar
   * burada görünmez (geçersiz kategori uyarı değil 422 üretir). Sıralama
   * anlamlıdır: modelin gerçekten kullandığı alanlar başta gelir.
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

export interface DefaultInfo {
  value: number | string;
  dtype: string;
  strategy: "median" | "mode";
}

export interface FeatureInfluenceEntry {
  split_count: number;
  gain: number;
  /** `split_count > 0` — model bu feature'ı gerçekten kullandı mı? */
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
  /** BEYAN: feature modele girdi olarak verildi. */
  used_as_model_feature: boolean;
  /** ÖLÇÜM: eğitilmiş ağaçlar onu gerçekten kullandı mı? */
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
    /** PII kolon ADLARI — sızıntı değil, "bunlar modele hiç girmedi" beyanı. */
    dropped_columns: string[];
    target: string;
  };
  feature_influence: FeatureInfluence;
  training_ranges: Record<string, TrainingRange>;
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
/* Hata                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * Backend'in 422 gövdesi.
 *
 * `input` ve `ctx` alanları BİLEREK yok: backend hatalı istekte gönderilen ham
 * değeri geri yansıtmıyor (sigorta talebi verisi log'lara/konsola sızmasın
 * diye). Elimizde yalnızca hangi alan (`loc`) ve ne yanlış (`msg`) var.
 */
export interface ValidationErrorItem {
  loc: string[];
  msg: string;
  type: string;
}

export interface ValidationErrorBody {
  detail: ValidationErrorItem[];
}
