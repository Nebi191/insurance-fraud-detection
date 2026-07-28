"""FastAPI application — `/predict`, `/health`, `/model-info`.

WHAT IT DOES, AND WHY IT DOES IT THIS WAY
-----------------------------------------

1) A THIN LAYER
   There is NO business logic here. The endpoints only (a) hand the validated
   request to `ModelBundle`, and (b) shape the result into the contract schema.
   Scoring, imputation and SHAP belong to `model.py`; input validation belongs to
   `schemas.py`. That keeps `model.py` testable without HTTP.

2) LOADED ONCE VIA `lifespan` (K1)
   The artifact is loaded once at application startup. If a file is missing or
   the metadata is inconsistent, `ModelBundle.load()` raises and the application
   does NOT start (fail-fast). A service that is "up but returns 500 to every
   request" is more dangerous than one that never starts, because it shows a
   green healthcheck.

3) THE `create_app()` FACTORY — FOR TESTABILITY
   The application is built inside a factory function; `app = create_app()` is
   merely a module-level instance (so the uvicorn target `app.main:app` still
   works).

   WHY IT MATTERS: the CORS origins used to be read once, at module import time.
   That made the wiring untestable — no test could catch a change that hard-coded
   the origin list, set `allow_methods=["*"]`, or turned on
   `allow_credentials=True`. With the factory, tests can build a FRESH
   application with different environment variables and verify the behaviour with
   a real `Origin`-carrying request. Phase 7's completion criterion depends
   directly on this path, and an untestable path is an untested path.

4) CORS: NO WILDCARDS, ENFORCED BY THIS CODE (K10)
   `allow_origins=["*"]` is never used. More importantly, that rule now applies
   on the environment-variable path too, not just to the DEFAULT: trying to
   deploy with `ALLOWED_ORIGINS="*"` fails at startup. Otherwise K10 would be a
   statement of intent only — and a security rule that a single environment
   variable can punch through is not a security rule.

5) THE ERROR BODY NEVER ECHOES CLIENT DATA
   Pydantic's default 422 body returns the submitted raw value in an `input`
   field. This API's body is insurance claim data; a bad request would reflect
   personal data into logs, reverse proxies and the browser console.
   `validation_exception_handler` drops the `input`/`ctx` fields and keeps `loc`
   and `msg` (so the frontend can still show per-field errors).

6) SYNCHRONOUS `def` ENDPOINTS — DELIBERATE
   Prediction is CPU-bound (LightGBM + SHAP, C extensions). Putting it inside an
   `async def` would block the event loop, and a single slow request would stall
   the whole server. Declaring them as synchronous `def` makes FastAPI run them
   in a thread pool, leaving the event loop free.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers

from app.model import (
    CALIBRATION_WARNING,
    RISK_RULE_DESCRIPTION,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    ModelBundle,
)
from app.schemas import (
    CalibrationInfo,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RiskThresholdsInfo,
)

# --------------------------------------------------------------------------- #
# CORS configuration
# --------------------------------------------------------------------------- #

ALLOWED_ORIGINS_ENV = "ALLOWED_ORIGINS"

# The Vite dev server. In Phase 7 the Netlify production origin (e.g.
# "https://<site>.netlify.app") will be added to the ALLOWED_ORIGINS environment
# variable — no code change, only an update to the variable on HF Spaces.
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173"


class CorsConfigurationError(ValueError):
    """The CORS configuration is unsafe — the application must not start."""


def get_allowed_origins() -> list[str]:
    """Parses the `ALLOWED_ORIGINS` environment variable into a comma-separated list.

    IT FAILS AT STARTUP ON TWO BAD CONFIGURATIONS:

      `ALLOWED_ORIGINS="*"`  -> the wildcard K10 forbids. Without this check the
          rule would only hold for the default value; a single environment
          variable would open the door to every site.
      `ALLOWED_ORIGINS=""`   -> the variable IS SET but resolves to no origin at
          all. This silently shuts out every browser client and is the hardest
          class of deploy bug to diagnose ("the API is up, the healthcheck is
          green, but the frontend cannot fetch anything"). If the variable is NOT
          SET at all, that is not an error — the default applies.
    """
    raw = os.getenv(ALLOWED_ORIGINS_ENV)
    explicitly_configured = raw is not None
    if raw is None:
        raw = DEFAULT_ALLOWED_ORIGINS

    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]

    wildcards = [origin for origin in origins if "*" in origin]
    if wildcards:
        raise CorsConfigurationError(
            f"{ALLOWED_ORIGINS_ENV} must not contain a wildcard (found: {wildcards}). "
            "Spell out every origin in full, e.g. "
            "ALLOWED_ORIGINS='http://localhost:5173,https://your-site.netlify.app'. "
            "If a subdomain pattern is required, use CORSMiddleware's "
            "allow_origin_regex parameter — allow_origins does not support globs."
        )

    if explicitly_configured and not origins:
        raise CorsConfigurationError(
            f"{ALLOWED_ORIGINS_ENV} is set but contains no valid origin "
            f"(raw value: {raw!r}). This would silently block every browser "
            "client. Provide at least one origin, or remove the "
            f"{ALLOWED_ORIGINS_ENV} variable entirely to use the default."
        )

    return origins


# --------------------------------------------------------------------------- #
# Request body size limit
# --------------------------------------------------------------------------- #

# A single claim JSON is under 2 KB. 64 KB leaves comfortable room for the field
# set to grow while keeping the server from pulling megabytes of body into
# memory. Starlette's default has no practical limit; for a public endpoint with
# no authentication that is a cheap DoS surface.
MAX_REQUEST_BODY_BYTES = 64 * 1024

# Methods that do NOT expect a body (Codex C-1).
#
# The streaming arm of the limit depended on the downstream application actually
# READING the body: the counter only advances when `receive()` is called.
# `/health` and `/model-info` never read the request body, so a client that sends
# no `Content-Length` (chunked) could stream an unbounded body at those endpoints
# and still get a 200 — measured: `GET /health` with a 160 KB chunked body -> 200.
#
# For these methods the middleware ITSELF consumes and counts the body. The cost
# is effectively zero: a normal GET has an empty body, so the loop ends on the
# first message.
BODYLESS_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "DELETE"})


class _BodyTooLarge(Exception):
    """The body limit was exceeded (used only inside the middleware)."""


class BodySizeLimitMiddleware:
    """Rejects requests whose body exceeds `max_body_bytes` with a 413.

    Written as raw ASGI middleware because `BaseHTTPMiddleware` already reads and
    buffers the body — applying the limit there would mean spending the very
    memory we are trying to protect first.

    TWO-STAGE PROTECTION:
      1. If `Content-Length` is present, the request is rejected before it is
         read at all (the cheap path).
      2. If the header is absent (chunked transfer encoding), body chunks are
         counted AS THEY STREAM and reading stops the moment the limit is
         exceeded. Trusting `Content-Length` alone would grant an unbounded body
         to any client that simply omits the header.
    """

    def __init__(self, app: Any, max_body_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_body_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                # Malformed Content-Length: do not trust the header, count the
                # stream instead.
                pass

        if scope.get("method", "").upper() in BODYLESS_METHODS:
            await self._call_with_drained_body(scope, receive, send)
            return

        received = 0
        exceeded = False
        handled = False

        async def counting_receive() -> MutableMapping[str, Any]:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    exceeded = True
                    # Cutting the read off here is the actual protection:
                    # otherwise the memory we are trying to save has already been
                    # spent.
                    raise _BodyTooLarge
            return message

        async def guarding_send(message: MutableMapping[str, Any]) -> None:
            """Replaces the application's response with a 413 once the limit is hit.

            WHY `except _BodyTooLarge` ALONE IS NOT ENOUGH:
            while parsing the body, FastAPI catches every error with
            `except Exception` and turns it into a 400 saying "There was an error
            parsing the body". Our exception was therefore swallowed before it
            ever reached us, and the client got a misleading 400. Checking the
            flag on the SEND path is the one reliable point that does not depend
            on the behaviour of any layer in between.
            """
            nonlocal handled
            if handled:
                return  # The 413 was sent; swallow the application's remaining messages.
            if exceeded:
                handled = True
                await self._reject(send)
                return
            await send(message)

        try:
            await self.app(scope, counting_receive, guarding_send)
        except _BodyTooLarge:
            # Reached only if no intermediate layer swallowed it.
            if not handled:
                handled = True
                await self._reject(send)

    async def _call_with_drained_body(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """For bodyless methods the middleware consumes and counts the body itself,
        then calls the application.

        WHY A SEPARATE PATH: in the normal flow the counter runs through
        `counting_receive` and only advances if the application reads the body.
        Because GET endpoints never read it, that arm never engaged at all (see
        `BODYLESS_METHODS`). Here we take the reading responsibility away from the
        application, which makes the limit INDEPENDENT of whether an endpoint
        cares about the body.

        If the limit is exceeded, reading stops right there — we never spend the
        memory we are trying to protect.
        """
        received = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # http.disconnect: the client is gone, there is no body left to count.
                break
            received += len(message.get("body", b""))
            if received > self.max_body_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        async def drained_receive() -> MutableMapping[str, Any]:
            """The body was consumed; if the application still reads, it sees empty."""
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, drained_receive, send)

    async def _reject(
        self, send: Callable[[MutableMapping[str, Any]], Awaitable[None]]
    ) -> None:
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "detail": (
                    f"Request body is too large. The limit is {self.max_body_bytes} bytes; "
                    "a single claim JSON is normally under 2 KB."
                )
            },
        )
        await response({"type": "http"}, _unused_receive, send)


async def _unused_receive() -> MutableMapping[str, Any]:  # pragma: no cover
    """Required by the `Response.__call__` signature; never called since no body is read."""
    return {"type": "http.disconnect"}


# --------------------------------------------------------------------------- #
# Validation error body
# --------------------------------------------------------------------------- #


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Strips the client's raw data out of the 422 body.

    KEPT: `loc` (which field), `msg` (what is wrong), `type` (machine-readable
    code). The frontend needs all three to show per-field errors; the status code
    and the structure of the `detail` list are preserved as well.

    DROPPED: `input` (the client's raw submitted value itself) and `ctx` (which
    carries input-derived data for some error types). The body sent to this API
    is insurance claim data; echoing it back on a bad request would carry that
    data into logs and the browser console.

    NOTE: the field name inside `loc` can originate from the client (on an
    unknown-field error). That is deliberate: only the KEY name comes back, never
    the VALUE, and naming the rejected field is essential for the error to be
    understandable.
    """
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    sanitized = [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": str(error.get("msg", "")),
            "type": str(error.get("type", "")),
        }
        for error in errors
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": sanitized},
    )


# --------------------------------------------------------------------------- #
# Application lifecycle
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the artifact once at startup; if it fails, the application does not start."""
    app.state.bundle = ModelBundle.load()
    yield
    # Nothing external to release on shutdown (no files or sockets are held);
    # letting the bundle be garbage collected is enough.


def get_bundle(request: Request) -> ModelBundle:
    """Dependency that hands the loaded artifact to the endpoints."""
    return request.app.state.bundle


BundleDep = Annotated[ModelBundle, Depends(get_bundle)]

# Routes are collected on a module-level router that the factory attaches with
# `include_router`. That keeps both decorator readability and factory
# flexibility.
router = APIRouter()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(bundle: BundleDep) -> HealthResponse:
    """Simple healthcheck.

    The `bundle` dependency is deliberate: since the application refuses to start
    without the artifact, reaching this point means "the model is loaded".
    `model_version` answers "which artifact is running?" in a single request
    after a deploy.
    """
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=bundle.metadata["model_version"],
    )


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest, bundle: BundleDep) -> PredictResponse:
    """Scores a single claim and explains it with SHAP contributions.

    `by_alias=True`: so the `capital-gains` / `capital-loss` fields come out with
    the pipeline's (hyphenated) column names. `model.py` assumes the keys are
    metadata column names; dumping without aliases would break that assumption.
    """
    result = bundle.predict(payload.model_dump(by_alias=True))
    return PredictResponse.model_validate(result)


# THE NOTE BELOW IS DELIBERATELY A COMMENT, NOT A DOCSTRING:
# An endpoint's docstring becomes the `description` in the OpenAPI schema, and
# `/openapi.json` is a public surface. Listing the names of the metadata keys we
# left OUT of the allowlist would carry the very internal structure we are trying
# to withhold into that public schema. The explanation stays here: just as useful
# to someone reading the source, invisible to OpenAPI.
#
#   metadata.json IS NOT RETURNED AS-IS. The sub-models in `schemas.py` declare
#   only the permitted fields; thanks to `extra="ignore"` everything else (the
#   preprocessing contract, hyperparameters, the source file name) is dropped. We
#   do not hand-write `dict.pop()` calls: a removal list gets forgotten somewhere
#   in the code, a schema cannot be — and `response_model` adds a second filter.
@router.get("/model-info", response_model=ModelInfoResponse, tags=["model card"])
def model_info(bundle: BundleDep) -> ModelInfoResponse:
    """Model card data — a projection of the fields cleared for publication."""
    metadata = bundle.metadata
    return ModelInfoResponse(
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        algorithm=metadata["algorithm"],
        trained_at=metadata["trained_at"],
        metrics=metadata["metrics"],
        dataset=metadata["dataset"],
        feature_list=metadata["feature_list"],
        feature_influence=metadata["feature_influence"],
        training_ranges=metadata["training_ranges"],
        defaults=metadata["defaults"],
        fairness=metadata["fairness"],
        library_versions=metadata["library_versions"],
        # The thresholds are read from the constants in `model.py` so the
        # frontend keeps no copy of its own and there is a single source of truth.
        risk_thresholds=RiskThresholdsInfo(
            low_below=RISK_THRESHOLD_MEDIUM,
            high_at_or_above=RISK_THRESHOLD_HIGH,
            rule=RISK_RULE_DESCRIPTION,
        ),
        probability_calibration=CalibrationInfo(
            calibrated=False,
            method="none",
            warning=CALIBRATION_WARNING,
        ),
    )


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    """Builds the application. CORS origins are read IN THIS CALL, not at import time.

    Tests can call this factory with different `ALLOWED_ORIGINS` values and verify
    the behaviour with a real HTTP request; the wiring is now observable.
    """
    application = FastAPI(
        title="Insurance Fraud Detection API",
        version="1.0.0",
        summary="LightGBM insurance claim fraud scoring, explained with SHAP.",
        description=(
            "Scores a single insurance claim and explains the decision with SHAP "
            "contributions.\n\n"
            "**Warning:** the probabilities are not calibrated, and the model uses "
            "protected attributes as features. See `GET /model-info` for details."
        ),
        lifespan=lifespan,
    )

    application.add_exception_handler(RequestValidationError, validation_exception_handler)

    # MIDDLEWARE ORDER MATTERS: `add_middleware` prepends to the list on every
    # call, so the LAST one added ends up outermost. We want CORS outermost so
    # that ALL responses — including the 413 — carry CORS headers; otherwise a
    # browser client cannot read the error and only sees an opaque network
    # failure.
    application.add_middleware(BodySizeLimitMiddleware, max_body_bytes=MAX_REQUEST_BODY_BYTES)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        # Only the methods and headers actually in use. Since the API performs no
        # authentication, `allow_credentials` stays off (turning it on would be a
        # pointless risk alongside the wildcard ban).
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    application.include_router(router)
    return application


# uvicorn target: `app.main:app`
app = create_app()
