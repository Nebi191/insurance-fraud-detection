/**
 * Backend istemcisi.
 *
 * NE YAPAR, NEDEN BÖYLE YAPAR
 * ---------------------------
 *
 * 1) TEK BİR HATA TÜRÜ
 *    Çağıran taraf üç ayrı başarısızlığı ayırt edebilmeli: ağ hatası,
 *    doğrulama hatası (422, alan bazlı), ve diğer sunucu hataları. `ApiError`
 *    bu üçünü tek tipte taşır ki UI `catch` içinde tahmin yürütmek zorunda
 *    kalmasın.
 *
 * 2) 422 GÖVDESİ ALAN BAZLI HATAYA ÇEVRİLİR
 *    Backend `loc: ["body", "witnesses"]` biçiminde dönüyor. Formun ihtiyacı
 *    olan şey `{ witnesses: "mesaj" }`; dönüşümü burada bir kez yapıyoruz ki
 *    her bileşen kendi ayrıştırmasını yazmasın.
 *
 * 3) YANIT ŞEKLİ ÇALIŞMA ZAMANINDA KONTROL EDİLİR
 *    TypeScript tipi bir İDDİADIR, garanti değil: `await response.json()`
 *    `any` döner ve yanlış şekildeki bir gövde uygulamanın derinlerinde
 *    anlamsız bir hatayla patlar. Sözleşmenin tutmadığı yerde ANINDA ve
 *    anlaşılır biçimde durmak, Faz 7'de backend/frontend sürümleri ayrışırsa
 *    teşhisi dakikalar yerine saniyeler alır.
 */

import type {
  HealthResponse,
  ModelInfoResponse,
  PredictRequest,
  PredictResponse,
} from "./types";

/**
 * Faz 7'de Netlify'da `VITE_API_URL` olarak HF Spaces URL'i verilecek.
 * Yerelde varsayılan uvicorn adresi kullanılır.
 */
export const API_BASE_URL = (
  import.meta.env["VITE_API_URL"] ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export type ApiErrorKind = "network" | "validation" | "server";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  /** Yalnızca `kind === "validation"` iken dolu: { alanAdı: mesaj }. */
  readonly fieldErrors: Record<string, string>;

  constructor(
    kind: ApiErrorKind,
    message: string,
    status: number | null = null,
    fieldErrors: Record<string, string> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * 422 gövdesini `{ alanAdı: mesaj }` haritasına çevirir.
 *
 * Gövde `unknown` olarak geziliyor, `ValidationErrorBody`'ye CAST EDİLMİYOR:
 * cast bir iddiadır, gelen gövde başka şekilde olsaydı hata form katmanının
 * derinlerinde anlamsız bir yerde patlardı. Burada her adım kontrol ediliyor
 * ve tanınmayan gövde sessizce boş haritaya düşüyor — kullanıcı yine genel
 * hata mesajını görür.
 */
function toFieldErrors(body: unknown): Record<string, string> {
  const errors: Record<string, string> = {};
  if (!isRecord(body)) return errors;

  const detail = body["detail"];
  if (!Array.isArray(detail)) return errors;

  for (const item of detail) {
    if (!isRecord(item)) continue;
    const location = item["loc"];
    const message = item["msg"];
    if (!Array.isArray(location) || typeof message !== "string") continue;

    // loc = ["body", "<alan>"] — ilk eleman gövdeyi işaret eder, sonuncuyu alıyoruz.
    const field = location[location.length - 1];
    if (typeof field === "string" && field !== "body" && !(field in errors)) {
      errors[field] = message;
    }
  }
  return errors;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (cause) {
    // Buraya CORS reddi de düşer ve tarayıcı gerekçeyi JS'e VERMEZ — mesajın
    // olası sebebi açıkça yazmasının nedeni bu.
    throw new ApiError(
      "network",
      `API'ye ulaşılamadı (${API_BASE_URL}). Backend çalışıyor mu? ` +
        "CORS reddi de bu hataya benzer görünür: backend'in ALLOWED_ORIGINS " +
        "değişkeni bu sayfanın origin'ini içermeli.",
      null,
      {},
    );
  }

  if (response.status === 422) {
    const body: unknown = await response.json().catch(() => null);
    throw new ApiError(
      "validation",
      "Girdi doğrulamadan geçmedi.",
      422,
      toFieldErrors(body),
    );
  }

  if (!response.ok) {
    throw new ApiError(
      "server",
      `Sunucu ${response.status} döndü.`,
      response.status,
    );
  }

  return (await response.json()) as T;
}

/** Yanıtın beklenen şekilde olduğunu çalışma zamanında doğrular. */
function assertShape(condition: boolean, what: string): void {
  if (!condition) {
    throw new ApiError(
      "server",
      `API yanıtı beklenen sözleşmeye uymuyor: ${what}. ` +
        "Backend ve frontend sürümleri ayrışmış olabilir.",
    );
  }
}

export async function predict(payload: PredictRequest): Promise<PredictResponse> {
  const body = await request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  assertShape(typeof body.fraud_probability === "number", "fraud_probability sayı değil");
  assertShape(Array.isArray(body.shap_values), "shap_values dizi değil");
  assertShape(
    Array.isArray(body.out_of_distribution_warnings),
    "out_of_distribution_warnings dizi değil",
  );
  return body;
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const body = await request<ModelInfoResponse>("/model-info");

  assertShape(isRecord(body.training_ranges), "training_ranges yok");
  assertShape(isRecord(body.defaults), "defaults yok");
  assertShape(
    isRecord(body.feature_influence?.features),
    "feature_influence.features yok",
  );
  assertShape(
    Array.isArray(body.feature_list?.pipeline_input_order),
    "feature_list.pipeline_input_order yok",
  );
  return body;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}
