/**
 * Backend client.
 *
 * WHAT IT DOES, AND WHY IT DOES IT THIS WAY
 * -----------------------------------------
 *
 * 1) ONE ERROR TYPE
 *    Callers need to distinguish three failures: a network error, a validation
 *    error (422, per field), and any other server error. `ApiError` carries all
 *    three in a single type so the UI never has to guess inside a `catch`.
 *
 * 2) THE 422 BODY IS TURNED INTO PER-FIELD ERRORS
 *    The backend returns `loc: ["body", "witnesses"]`. What the form needs is
 *    `{ witnesses: "message" }`; the conversion happens here once so that no
 *    component has to write its own parsing.
 *
 * 3) RESPONSE SHAPES ARE CHECKED AT RUNTIME
 *    A TypeScript type is an ASSERTION, not a guarantee: `await response.json()`
 *    returns `any`, and a wrongly-shaped body blows up deep inside the app with
 *    a meaningless error. Stopping IMMEDIATELY and legibly at the point the
 *    contract breaks turns a Phase 7 backend/frontend version skew from a
 *    minutes-long diagnosis into a seconds-long one.
 */

import type {
  HealthResponse,
  ModelInfoResponse,
  PredictRequest,
  PredictResponse,
} from "./types";

/**
 * In Phase 7 Netlify supplies the Hugging Face Spaces URL via `VITE_API_URL`.
 * Locally we fall back to the default uvicorn address.
 */
export const API_BASE_URL = (
  import.meta.env["VITE_API_URL"] ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export type ApiErrorKind = "network" | "validation" | "server";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  /** Populated only when `kind === "validation"`: { fieldName: message }. */
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
 * Converts a 422 body into a `{ fieldName: message }` map.
 *
 * The body is walked as `unknown` and NEVER cast to `ValidationErrorBody`: a
 * cast is an assertion, and a differently-shaped body would then explode
 * somewhere meaningless inside the form layer. Every step is checked here, and
 * an unrecognised body degrades silently to an empty map — the user still sees
 * the generic error message.
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

    // loc = ["body", "<field>"] — the first element points at the body, so we
    // take the last one. `location` was already narrowed to `unknown[]` by
    // `Array.isArray` above; the value read out is verified at runtime right
    // below with `typeof field === "string"`.
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
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
  } catch {
    // A CORS rejection also lands here, and the browser deliberately withholds
    // the reason from JS — which is why the message spells out the likely cause.
    throw new ApiError(
      "network",
      `Could not reach the API (${API_BASE_URL}). Is the backend running? ` +
        "A CORS rejection looks the same from here: the backend's ALLOWED_ORIGINS " +
        "must include this page's origin.",
      null,
      {},
    );
  }

  if (response.status === 422) {
    const body: unknown = await response.json().catch(() => null);
    throw new ApiError(
      "validation",
      "The input did not pass validation.",
      422,
      toFieldErrors(body),
    );
  }

  if (!response.ok) {
    throw new ApiError(
      "server",
      `The server returned ${response.status}.`,
      response.status,
    );
  }

  return (await response.json()) as T;
}

/** Verifies at runtime that a response really has the expected shape. */
function assertShape(condition: boolean, what: string): void {
  if (!condition) {
    throw new ApiError(
      "server",
      `The API response does not match the expected contract: ${what}. ` +
        "The backend and frontend versions may have drifted apart.",
    );
  }
}

export async function predict(payload: PredictRequest): Promise<PredictResponse> {
  const body = await request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  assertShape(typeof body.fraud_probability === "number", "fraud_probability is not a number");
  assertShape(Array.isArray(body.shap_values), "shap_values is not an array");
  assertShape(
    Array.isArray(body.out_of_distribution_warnings),
    "out_of_distribution_warnings is not an array",
  );
  return body;
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const body = await request<ModelInfoResponse>("/model-info");

  assertShape(isRecord(body.training_ranges), "training_ranges is missing");
  assertShape(isRecord(body.physical_ranges), "physical_ranges is missing");
  assertShape(isRecord(body.defaults), "defaults is missing");
  assertShape(
    isRecord(body.feature_influence?.features),
    "feature_influence.features is missing",
  );
  assertShape(
    Array.isArray(body.feature_list?.pipeline_input_order),
    "feature_list.pipeline_input_order is missing",
  );
  return body;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}
