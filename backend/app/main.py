"""FastAPI uygulaması — `/predict`, `/health`, `/model-info`.

NE YAPAR, NEDEN BÖYLE YAPAR
---------------------------

1) İNCE KATMAN
   Burada iş mantığı YOKTUR. Endpoint'ler yalnızca (a) doğrulanmış isteği
   `ModelBundle`'a verir, (b) sonucu sözleşmedeki şemaya sokar. Skorlama,
   imputasyon ve SHAP `model.py`'ın; girdi doğrulaması `schemas.py`'ın işi.
   Böylece `model.py` HTTP olmadan da test edilebilir kalır.

2) `lifespan` İLE TEK SEFERLİK YÜKLEME (K1)
   Artefakt uygulama açılışında bir kez yüklenir. Dosya yoksa veya metadata
   tutarsızsa `ModelBundle.load()` fırlatır ve uygulama HİÇ açılmaz
   (fail-fast). "Ayakta ama her isteğe 500 dönen" bir servis, healthcheck'i
   yeşil gösterdiği için açılmayandan daha tehlikelidir.

3) `create_app()` FABRİKASI — TEST EDİLEBİLİRLİK İÇİN
   Uygulama bir fabrika fonksiyonunda kurulur; `app = create_app()` yalnızca
   modül düzeyinde bir örnektir (uvicorn hedefi `app.main:app` bozulmaz).

   NEDEN ÖNEMLİ: eskiden CORS origin'leri modül import edilirken bir kez
   okunuyordu. Bu, kablolamanın test edilemez olması demekti — origin listesini
   sabit kodlayan, `allow_methods=["*"]` yapan ya da `allow_credentials=True`
   açan bir değişikliği hiçbir test yakalayamıyordu. Fabrika sayesinde testler
   farklı ortam değişkenleriyle YENİ bir uygulama kurup gerçek `Origin`
   başlıklı istekle davranışı doğrulayabiliyor. Faz 7'nin bitti kriteri
   doğrudan bu yola bağlı; test edilemeyen yol, test edilmemiş yoldur.

4) CORS: WILDCARD YASAK VE BU KOD TARAFINDAN ZORLANIR (K10)
   `allow_origins=["*"]` kullanılmaz. Daha da önemlisi: bu kural artık yalnızca
   VARSAYILAN değerde değil, ortam değişkeni yolunda da geçerli.
   `ALLOWED_ORIGINS="*"` ile deploy etmeye çalışmak açılışta patlar. Aksi hâlde
   K10 sadece bir niyet beyanı olurdu; tek bir ortam değişkeniyle delinebilen
   bir güvenlik kuralı, güvenlik kuralı değildir.

5) HATA GÖVDESİ İSTEMCİ VERİSİNİ GERİ YANSITMAZ
   Pydantic'in varsayılan 422 gövdesi `input` alanında gönderilen ham değeri
   geri yollar. Bu API'nin gövdesi sigorta talebi verisidir; hatalı bir istek
   log'lara, ters proxy'lere ve tarayıcı konsoluna kişisel veri yansıtırdı.
   `validation_exception_handler` `input`/`ctx` alanlarını düşürür, `loc` ve
   `msg`'i korur (frontend hâlâ alan bazlı hata gösterebilir).

6) SENKRON `def` ENDPOINT'LER — BİLİNÇLİ
   Tahmin CPU-bound'dur (LightGBM + SHAP, C uzantısı). `async def` içine
   koysaydık event loop'u bloklardı ve tek yavaş istek tüm sunucuyu
   durdururdu. Senkron `def` tanımlayınca FastAPI bunları thread havuzunda
   koşturur; event loop serbest kalır.
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
# CORS yapılandırması
# --------------------------------------------------------------------------- #

ALLOWED_ORIGINS_ENV = "ALLOWED_ORIGINS"

# Vite dev sunucusu. Faz 7'de Netlify production origin'i (ör.
# "https://<site>.netlify.app") ALLOWED_ORIGINS ortam değişkenine eklenecek —
# kod değişmeyecek, sadece HF Spaces'teki değişken güncellenecek.
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173"


class CorsConfigurationError(ValueError):
    """CORS yapılandırması güvenli değil — uygulama açılmamalı."""


def get_allowed_origins() -> list[str]:
    """`ALLOWED_ORIGINS` ortam değişkenini virgülle ayrılmış listeye çevirir.

    İKİ HATALI YAPILANDIRMAYI AÇILIŞTA PATLATIR:

      `ALLOWED_ORIGINS="*"`  -> K10'un yasakladığı wildcard. Buradaki kontrol
          olmadan kural sadece varsayılan değerde geçerli olurdu; tek bir ortam
          değişkeniyle her siteye kapı açılırdı.
      `ALLOWED_ORIGINS=""`   -> değişken SET EDİLMİŞ ama hiçbir origin
          çözülmüyor. Sessizce tüm tarayıcı istemcilerini kapatır ve deploy'da
          teşhisi en zor hata türüdür ("API ayakta, healthcheck yeşil, ama
          frontend hiçbir şey çekemiyor"). Değişken hiç SET EDİLMEMİŞSE bu
          hata değildir — varsayılan devreye girer.
    """
    raw = os.getenv(ALLOWED_ORIGINS_ENV)
    explicitly_configured = raw is not None
    if raw is None:
        raw = DEFAULT_ALLOWED_ORIGINS

    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]

    wildcards = [origin for origin in origins if "*" in origin]
    if wildcards:
        raise CorsConfigurationError(
            f"{ALLOWED_ORIGINS_ENV} wildcard içeremez (bulunan: {wildcards}). "
            "Her origin'i tam olarak yazın, ör. "
            "ALLOWED_ORIGINS='http://localhost:5173,https://siteniz.netlify.app'. "
            "Alt alan adı deseni gerekiyorsa CORSMiddleware'in allow_origin_regex "
            "parametresi kullanılmalıdır — allow_origins glob desteklemez."
        )

    if explicitly_configured and not origins:
        raise CorsConfigurationError(
            f"{ALLOWED_ORIGINS_ENV} tanımlı ama hiçbir geçerli origin içermiyor "
            f"(ham değer: {raw!r}). Bu, bütün tarayıcı istemcilerini sessizce "
            "engellerdi. En az bir origin yazın ya da varsayılanı kullanmak için "
            f"{ALLOWED_ORIGINS_ENV} değişkenini tamamen kaldırın."
        )

    return origins


# --------------------------------------------------------------------------- #
# İstek gövdesi boyut sınırı
# --------------------------------------------------------------------------- #

# Tek bir talep JSON'u 2 KB'nin altında. 64 KB, olası bir alan genişlemesine
# rahat yer bırakırken sunucuyu megabaytlarca gövdeyi belleğe almaktan korur.
# Starlette'in varsayılanında pratik bir sınır yoktur; kimlik doğrulaması
# olmayan public bir endpoint için bu ucuz bir DoS yüzeyidir.
MAX_REQUEST_BODY_BYTES = 64 * 1024


class _BodyTooLarge(Exception):
    """Gövde sınırı aşıldı (yalnızca middleware içinde kullanılır)."""


class BodySizeLimitMiddleware:
    """Gövdesi `max_body_bytes`'ı aşan istekleri 413 ile reddeder.

    Ham ASGI middleware'i olarak yazıldı çünkü `BaseHTTPMiddleware` gövdeyi
    zaten okuyup tamponlar — sınırı orada uygulamak, korumaya çalıştığımız
    belleği önce harcamak olurdu.

    İKİ AŞAMALI KORUMA:
      1. `Content-Length` varsa istek daha okunmadan reddedilir (ucuz yol).
      2. Başlık yoksa (chunked transfer encoding) gövde parçaları AKARKEN
         sayılır ve sınır aşılır aşılmaz durdurulur. Sadece `Content-Length`'e
         güvenmek, başlığı hiç göndermeyen bir istemciye sınırsız gövde hakkı
         tanımak demekti.
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
                # Bozuk Content-Length: başlığa güvenmeyip akışı sayarak ilerle.
                pass

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
                    # Okumayı burada kesmek asıl korumadır: aksi hâlde tam da
                    # engellemeye çalıştığımız bellek zaten harcanmış olurdu.
                    raise _BodyTooLarge
            return message

        async def guarding_send(message: MutableMapping[str, Any]) -> None:
            """Sınır aşıldıysa uygulamanın ürettiği yanıtı 413 ile değiştirir.

            NEDEN SADECE `except _BodyTooLarge` YETMİYOR:
            FastAPI gövdeyi ayrıştırırken `except Exception` ile her hatayı
            yakalayıp "There was an error parsing the body" diyen bir 400'e
            çeviriyor. Yani istisnamız bize hiç ulaşmadan yutuluyordu ve
            istemci yanıltıcı bir 400 alıyordu. Bayrağı GÖNDERİM yolunda
            kontrol etmek, aradaki hiçbir katmanın davranışına bağlı olmayan
            tek güvenilir nokta.
            """
            nonlocal handled
            if handled:
                return  # 413 gönderildi; uygulamanın kalan mesajlarını yut.
            if exceeded:
                handled = True
                await self._reject(send)
                return
            await send(message)

        try:
            await self.app(scope, counting_receive, guarding_send)
        except _BodyTooLarge:
            # Hiçbir ara katman yutmadıysa buraya düşer.
            if not handled:
                handled = True
                await self._reject(send)

    async def _reject(
        self, send: Callable[[MutableMapping[str, Any]], Awaitable[None]]
    ) -> None:
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "detail": (
                    f"İstek gövdesi çok büyük. Üst sınır {self.max_body_bytes} bayt; "
                    "tek bir talep JSON'u normalde 2 KB'nin altındadır."
                )
            },
        )
        await response({"type": "http"}, _unused_receive, send)


async def _unused_receive() -> MutableMapping[str, Any]:  # pragma: no cover
    """`Response.__call__` imzası gereği; gövde okunmayacağı için çağrılmaz."""
    return {"type": "http.disconnect"}


# --------------------------------------------------------------------------- #
# Doğrulama hatası gövdesi
# --------------------------------------------------------------------------- #


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """422 gövdesinden istemcinin gönderdiği ham veriyi ayıklar.

    KORUNAN: `loc` (hangi alan), `msg` (ne yanlış), `type` (makine-okunur kod).
    Frontend alan bazlı hata gösterebilmek için bu üçüne ihtiyaç duyar; durum
    kodu ve `detail` listesi yapısı da aynen korunur.

    DÜŞÜRÜLEN: `input` (istemcinin gönderdiği ham değerin ta kendisi) ve `ctx`
    (bazı hata tiplerinde girdi türevleri taşır). Bu API'ye gönderilen gövde
    sigorta talebi verisidir; hatalı bir istekte onu geri yansıtmak veriyi
    log'lara ve tarayıcı konsoluna taşır.

    NOT: `loc` içindeki alan adı istemciden gelebilir (bilinmeyen alan
    hatasında). Bu bilinçli: DEĞER değil yalnızca ANAHTAR adı geri döner ve
    hangi alanın reddedildiğini söylemek hatanın anlaşılması için şarttır.
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
# Uygulama yaşam döngüsü
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Artefaktı açılışta bir kez yükler; başarısız olursa uygulama açılmaz."""
    app.state.bundle = ModelBundle.load()
    yield
    # Kapanışta serbest bırakılacak dış kaynak yok (dosya/soket tutulmuyor);
    # bundle'ı GC'ye bırakmak yeterli.


def get_bundle(request: Request) -> ModelBundle:
    """Yüklenmiş artefaktı endpoint'lere veren bağımlılık."""
    return request.app.state.bundle


BundleDep = Annotated[ModelBundle, Depends(get_bundle)]

# Rotalar modül düzeyinde bir router'da toplanır; fabrika bunu `include_router`
# ile bağlar. Böylece hem dekoratör okunabilirliği hem fabrika esnekliği korunur.
router = APIRouter()


# --------------------------------------------------------------------------- #
# Endpoint'ler
# --------------------------------------------------------------------------- #


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(bundle: BundleDep) -> HealthResponse:
    """Basit healthcheck.

    `bundle` bağımlılığı bilinçli: uygulama zaten artefakt olmadan açılmadığı
    için buraya gelmek "model yüklü" demektir. `model_version` deploy sonrası
    "hangi artefakt koşuyor?" sorusunu tek istekle cevaplar.
    """
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=bundle.metadata["model_version"],
    )


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest, bundle: BundleDep) -> PredictResponse:
    """Tek bir talebi skorlar ve SHAP katkılarıyla açıklar.

    `by_alias=True`: `capital-gains` / `capital-loss` alanları pipeline'ın
    kolon adlarıyla (tireli) çıksın diye. `model.py` anahtarların metadata
    kolon adları olduğunu varsayar; alias'sız dump o varsayımı bozardı.
    """
    result = bundle.predict(payload.model_dump(by_alias=True))
    return PredictResponse.model_validate(result)


# AŞAĞIDAKİ AÇIKLAMA BİLİNÇLİ OLARAK DOCSTRING DEĞİL, YORUM:
# Bir endpoint'in docstring'i OpenAPI şemasına `description` olarak girer ve
# `/openapi.json` public bir yüzeydir. Beyaz listenin DIŞINDA bıraktığımız
# metadata anahtarlarının adlarını docstring'de saymak, tam da gizlemeye
# çalıştığımız iç yapıyı public şemaya taşırdı. Açıklama burada duruyor:
# kaynağı okuyan için değeri aynı, OpenAPI için görünmez.
#
#   metadata.json OLDUĞU GİBİ DÖNDÜRÜLMEZ. `schemas.py`'daki alt modeller
#   yalnızca izin verilen alanları ilan eder; `extra="ignore"` sayesinde geri
#   kalan her şey (ön işleme sözleşmesi, hiperparametreler, kaynak dosya adı)
#   düşer. Elle `dict.pop()` yapmıyoruz: kırpma listesi kodun bir yerinde
#   unutulur, şema ise unutulamaz — ayrıca `response_model` ikinci bir süzgeç
#   kurar.
@router.get("/model-info", response_model=ModelInfoResponse, tags=["model card"])
def model_info(bundle: BundleDep) -> ModelInfoResponse:
    """Model card verisi — yayınlanmasına izin verilen alanların projeksiyonu."""
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
        # Eşikler `model.py`'daki sabitlerden okunur: frontend kendi kopyasını
        # tutmasın, tek doğruluk kaynağı olsun.
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
# Uygulama fabrikası
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    """Uygulamayı kurar. CORS origin'leri BU ÇAĞRIDA okunur, import anında değil.

    Testler bu fabrikayı farklı `ALLOWED_ORIGINS` değerleriyle çağırıp gerçek
    HTTP isteğiyle davranışı doğrulayabilir; kablolama artık gözlemlenebilir.
    """
    application = FastAPI(
        title="Insurance Fraud Detection API",
        version="1.0.0",
        summary="LightGBM tabanlı sigorta talebi dolandırıcılık skorlaması (SHAP açıklamalı).",
        description=(
            "Tek bir sigorta talebini skorlar ve kararı SHAP katkılarıyla açıklar.\n\n"
            "**Uyarı:** olasılıklar kalibre değildir ve model korunan nitelikleri "
            "feature olarak kullanır. Ayrıntı için `GET /model-info`."
        ),
        lifespan=lifespan,
    )

    application.add_exception_handler(RequestValidationError, validation_exception_handler)

    # MIDDLEWARE SIRASI ÖNEMLİ: `add_middleware` her çağrıda listenin BAŞINA
    # ekler, yani SON eklenen en dışta kalır. CORS'u en dışta istiyoruz ki 413
    # dâhil TÜM yanıtlar CORS başlıklarını alsın — aksi hâlde tarayıcı istemcisi
    # hatayı okuyamaz, yalnızca opak bir ağ hatası görür.
    application.add_middleware(BodySizeLimitMiddleware, max_body_bytes=MAX_REQUEST_BODY_BYTES)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        # Sadece gerçekten kullanılan metod/başlıklar. API kimlik doğrulaması
        # yapmadığı için `allow_credentials` açılmaz (açmak, wildcard yasağıyla
        # birlikte anlamsız bir risk olurdu).
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    application.include_router(router)
    return application


# uvicorn hedefi: `app.main:app`
app = create_app()
