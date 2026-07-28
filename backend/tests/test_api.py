"""Faz 1 API testleri.

TEST FELSEFESİ
--------------
Bu dosya "endpoint 200 döndü mü" testinden fazlasını yapmaya çalışır. Her test
somut bir SESSİZ HATA sınıfını kapatır:

  * Şema ile artefakt birbirinden ayrışabilir  -> Literal/metadata senkron testi
  * `"?"` NaN'a çevrilmezse encoder onu -1'e kodlar ve imputer devreye girmez;
    olasılık DEĞİŞMEYEBİLİR, yani uçtan uca test bunu yakalayamaz
                                               -> encoder seviyesinde test
  * SHAP'te yanlış sınıf ekseni seçilirse tüm işaretler ters döner ama yanıt
    yine "geçerli" görünür                     -> toplanabilirlik testi
  * Yanıt şeması genişlerse PII/hiperparametre sızabilir
                                               -> beyaz liste + PII testleri
"""

from __future__ import annotations

import copy
import json
import logging
import math
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any, get_args

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import model as model_module
from app.main import (
    ALLOWED_ORIGINS_ENV,
    DEFAULT_ALLOWED_ORIGINS,
    MAX_REQUEST_BODY_BYTES,
    CorsConfigurationError,
    app,
    create_app,
    get_allowed_origins,
    model_info,
)
from app.model import (
    METADATA_PATH,
    MODELS_DIR,
    PIPELINE_PATH,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    ArtifactError,
    ModelBundle,
    _positive_class_index,
    classify_risk,
)
from app.schemas import PREDICT_REQUEST_EXAMPLE, PredictRequest

# `train_pipeline.py` bu kolonları modele hiç sokmaz. İsimleri burada tek
# yerde tutuyoruz ki sızıntı testleri tek noktadan güncellensin.
PII_COLUMNS = ("policy_number", "insured_zip", "incident_location")

# `/model-info` yanıtında ASLA görünmemesi gereken metadata anahtarları.
FORBIDDEN_MODEL_INFO_KEYS = ("preprocessing_contract", "model_params", "source_file")


def _fold(text: str) -> str:
    """Türkçe metni aksan/nokta farklarından bağımsız aranabilir hâle getirir.

    NEDEN GEREKLİ: `"DENETİMİ".lower()` Python'da `"deneti̇mi̇"` üretir — Türkçe
    noktalı `İ`, `i` + U+0307 (birleşen nokta) olarak çözülür. Bu yüzden düz bir
    `"denetim" in text.lower()` kontrolü metin doğru olmasına rağmen başarısız
    olur. NFKD ayrıştırması + birleşen işaretlerin atılması bu tuzağı kapatır ve
    testi ç/ş/ı gibi harflere de dayanıklı kılar.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.translate(str.maketrans("çğıöşü", "cgiosu"))


# --------------------------------------------------------------------------- #
# 1) Artefakt yükleme
# --------------------------------------------------------------------------- #


def test_artifacts_actually_load(bundle: ModelBundle) -> None:
    """pipeline.pkl ve metadata.json gerçekten yüklendi ve tutarlı."""
    assert bundle.pipeline is not None
    assert bundle.explainer is not None
    assert len(bundle.input_order) == 34
    assert set(bundle.defaults) == set(bundle.input_order)
    # `ModelBundle.load()` içindeki doğrulamalar (sözleşme, feature hizası,
    # SHAP toplanabilirliği) patlamadıysa buraya gelinebilmiş demektir.


def test_pipeline_is_a_single_sklearn_pipeline(bundle: ModelBundle) -> None:
    """Tek Pipeline kuralı: preprocessing modelin ayrılmaz parçası."""
    assert list(bundle.pipeline.named_steps) == ["preprocessor", "model"]


def test_artifact_path_is_derived_from_the_package_not_from_input(bundle: ModelBundle) -> None:
    """K1: artefakt yolu sabit ve paket konumundan türetilir.

    Pickle yüklemek kod çalıştırmakla eşdeğerdir; yolun istekten/ortamdan
    gelmesi keyfi kod çalıştırma demek olurdu.
    """
    assert MODELS_DIR.name == "models"
    assert MODELS_DIR.parent.name == "backend"
    assert PIPELINE_PATH.parent == MODELS_DIR
    assert METADATA_PATH.parent == MODELS_DIR


def test_missing_artifact_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Artefakt yoksa `ModelBundle.load()` net bir hata fırlatır."""
    monkeypatch.setattr(model_module, "PIPELINE_PATH", tmp_path / "yok.pkl")
    with pytest.raises(model_module.ArtifactError, match="bulunamadı"):
        ModelBundle.load()


def test_app_refuses_to_start_without_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """K1 fail-fast: metadata yoksa uygulama HİÇ açılmaz.

    "Ayakta ama her isteğe 500 dönen" bir servis, healthcheck'i yeşil
    gösterdiği için açılmayan bir servisten daha tehlikelidir.
    """
    monkeypatch.setattr(model_module, "METADATA_PATH", tmp_path / "yok.json")
    with pytest.raises(model_module.ArtifactError), TestClient(app):
        pass  # pragma: no cover - buraya gelinmemeli


# --------------------------------------------------------------------------- #
# 2) /health
# --------------------------------------------------------------------------- #


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


# --------------------------------------------------------------------------- #
# 3) /model-info — beyaz liste ve sızıntı
# --------------------------------------------------------------------------- #


def test_model_info_returns_model_card_fields(client: TestClient, metadata: dict[str, Any]) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()

    assert body["model_version"] == metadata["model_version"]
    assert body["metrics"]["test_pr_auc"] == pytest.approx(metadata["metrics"]["test_pr_auc"])
    assert body["feature_list"]["pipeline_input_order"] == metadata["feature_list"]["pipeline_input_order"]
    assert body["trained_at"] == metadata["trained_at"]
    # K6: eşikler ve kalibrasyon uyarısı yayınlanıyor.
    assert body["risk_thresholds"]["low_below"] == RISK_THRESHOLD_MEDIUM
    assert body["risk_thresholds"]["high_at_or_above"] == RISK_THRESHOLD_HIGH
    assert body["probability_calibration"]["calibrated"] is False
    assert "kalibre" in body["probability_calibration"]["warning"].lower()


def test_model_info_does_not_leak_internal_keys(client: TestClient) -> None:
    """preprocessing_contract / model_params / source_file yanıtta OLMAMALI.

    Sadece üst seviye anahtarlara değil, serileşmiş METNİN tamamına bakıyoruz:
    iç içe bir yerde sızarsa da yakalansın.
    """
    response = client.get("/model-info")
    body = response.json()

    assert "preprocessing_contract" not in body
    assert "model_params" not in body
    assert "source_file" not in body["dataset"]

    raw = response.text
    for key in FORBIDDEN_MODEL_INFO_KEYS:
        assert key not in raw, f"'{key}' /model-info yanıtına sızmış"

    # Hiperparametre değerleri de sızmamalı (model_params kaldırıldı ama
    # değerlerin başka bir alandan dolaylı çıkmadığını da doğrulayalım).
    assert "n_estimators" not in raw
    assert "learning_rate" not in raw


def test_model_info_pii_appears_only_as_dropped_column_declaration(client: TestClient) -> None:
    """PII kolon ADLARI yalnızca "bunları attık" beyanında geçebilir.

    NOT — sözleşmede bir gerilim var ve bilinçli olarak böyle çözüldü:
    K9 `feature_list.dropped_columns` alanını beyaz listeye alıyor, ama o liste
    tanımı gereği `policy_number` / `insured_zip` / `incident_location`
    adlarını İÇERİR. Bu bir sızıntı değildir — PII *değeri* değil, "bu kolonlar
    modele hiç girmedi" beyanıdır ve model card için olumlu bir bilgidir.
    Test bu yüzden şunu doğrular: bu adlar dropped_columns DIŞINDA hiçbir yerde
    geçmiyor.
    """
    body = client.get("/model-info").json()

    dropped = body["feature_list"].pop("dropped_columns")
    assert set(PII_COLUMNS).issubset(set(dropped)), "PII kolonları drop listesinde olmalı"

    remainder = json.dumps(body, ensure_ascii=False)
    for column in PII_COLUMNS:
        assert column not in remainder, f"'{column}' dropped_columns dışında da geçiyor"

    # PII kolonları modele hiç girmediği için varsayılan/aralık da üretemez.
    for column in PII_COLUMNS:
        assert column not in body["defaults"]
        assert column not in body["training_ranges"]
        assert column not in body["feature_list"]["pipeline_input_order"]


# --------------------------------------------------------------------------- #
# 4) /predict — happy path
# --------------------------------------------------------------------------- #


def test_predict_with_claude_md_example(client: TestClient) -> None:
    """CLAUDE.md'deki örnek istek BİREBİR çalışıyor."""
    response = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
    assert response.status_code == 200, response.text

    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["risk_level"] == classify_risk(body["fraud_probability"])


def test_predict_response_has_exactly_the_contract_fields(client: TestClient) -> None:
    """Sözleşmedeki 4 alan, fazlası yok (sızıntı yüzeyi dar kalsın)."""
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert set(body) == {
        "fraud_probability",
        "risk_level",
        "shap_values",
        "out_of_distribution_warnings",
    }


def test_predict_with_empty_body_uses_defaults(client: TestClient) -> None:
    """`{}` ile de çalışır: 34 alanın hepsi metadata.defaults'tan dolar."""
    response = client.post("/predict", json={})
    assert response.status_code == 200, response.text
    assert 0.0 <= response.json()["fraud_probability"] <= 1.0


def test_predict_is_deterministic(client: TestClient) -> None:
    """Aynı girdi -> aynı olasılık (bit bazında)."""
    first = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    second = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert first == second


def test_out_of_distribution_warnings_fire_end_to_end(client: TestClient) -> None:
    """Faz 2 bitti kriteri: eğitim dışı değer `/predict` üzerinden uyarı üretir.

    Faz 1'de bu test "her zaman boş" diyordu ve bilinçli olarak Faz 2'de
    güncelleneceği yazılmıştı. Güncellenen hâli sözleşmenin iki ucunu da bağlar:
    aralık İÇİNDEKİ istek uyarı üretmemeli, DIŞINDAKİ üretmeli.
    """
    # CLAUDE.md örneği tamamen eğitim aralığının içinde.
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert body["out_of_distribution_warnings"] == []

    # witnesses eğitimde 0-3, age 20-64. İkisi de fiziksel olarak mümkün,
    # yani Pydantic kabul eder — ama model bunları hiç görmedi.
    body = client.post("/predict", json={"witnesses": 9, "age": 110}).json()
    assert set(body["out_of_distribution_warnings"]) == {"witnesses", "age"}
    # Uyarı tahmini ENGELLEMEZ: skor yine döner.
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}


# --------------------------------------------------------------------------- #
# 5) /predict — doğrulama reddi (422)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"incident_severity": "Katastrofik"}, "geçersiz Literal"),
        ({"policy_state": "TX"}, "eğitimde olmayan kategori"),
        ({"collision_type": "??"}, "'?' dışında bilinmeyen işaret"),
        ({"witnesses": 999}, "sayısal üst sınır"),
        ({"witnesses": -1}, "sayısal alt sınır"),
        ({"age": 5}, "fiziksel alt sınır"),
        ({"auto_year": 3000}, "fiziksel üst sınır"),
        ({"capital-gains": -1}, "kazanç negatif olamaz"),
        ({"capital-loss": 100}, "zarar bu veri setinde pozitif olamaz"),
        ({"total_claim_amount": -5}, "tutar negatif olamaz"),
        ({"number_of_vehicles_involved": 0}, "en az 1 araç"),
        ({"incident_hour_of_the_day": 24}, "gün 0-23 saat"),
        ({"unknown_field": 1}, "extra='forbid'"),
        ({"capital_gains": 1}, "alt tire yazım — alias 'capital-gains' olmalı"),
        ({"incident_date": "2015-01-25"}, "ham tarih API yüzeyinde yok (K2)"),
        ({"policy_bind_date": "1990-01-01"}, "ham tarih API yüzeyinde yok (K2)"),
        ({"policy_number": 521585}, "PII alanı kabul edilmez"),
        ({"insured_zip": 466132}, "PII alanı kabul edilmez"),
        ({"witnesses": "iki"}, "tip hatası"),
    ],
)
def test_predict_rejects_invalid_input(client: TestClient, payload: dict, reason: str) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"{reason} reddedilmeliydi: {response.text}"


def test_predict_accepts_question_mark(client: TestClient) -> None:
    """'?' üç alanda geçerli bir değerdir ve 200 döner."""
    response = client.post(
        "/predict",
        json={"collision_type": "?", "property_damage": "?", "police_report_available": "?"},
    )
    assert response.status_code == 200, response.text
    assert 0.0 <= response.json()["fraud_probability"] <= 1.0


# --------------------------------------------------------------------------- #
# 6) preprocessing_contract adımları GERÇEKTEN uygulanıyor mu?
# --------------------------------------------------------------------------- #


def test_question_mark_becomes_nan_not_an_unknown_category(bundle: ModelBundle) -> None:
    """`question_mark_to_nan` adımının ASIL testi.

    NEDEN UÇTAN UCA DEĞİL DE ENCODER SEVİYESİNDE:
    Bu modelde `collision_type` / `property_damage` / `police_report_available`
    feature'larının split sayısı SIFIRDIR (LightGBM hiç kullanmıyor). Yani "?"
    yanlışlıkla -1'e kodlansa bile `fraud_probability` DEĞİŞMEZ ve uçtan uca
    bir test bu hatayı asla yakalayamaz. Sözleşme ihlali sessizdir.

    O yüzden doğrudan pipeline'ın gördüğü kodlanmış değere bakıyoruz:
    doğru davranışta imputer'ın mod değerinin kodu, yanlış davranışta -1.
    """
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    cat_step = preprocessor.named_transformers_["cat"]
    encoder = cat_step.named_steps["encoder"]
    imputer = cat_step.named_steps["imputer"]
    cat_columns = list(preprocessor.transformers_[0][2])

    for column in bundle.question_mark_columns:
        frame = bundle.prepare_row({column: "?"})
        assert frame[column].isna().all(), f"'{column}' için '?' NaN'a çevrilmedi"

        index = cat_columns.index(column)
        expected_category = imputer.statistics_[index]
        expected_code = float(list(encoder.categories_[index]).index(expected_category))

        encoded = float(preprocessor.transform(frame)[f"cat__{column}"].iloc[0])
        assert encoded != -1.0, f"'{column}': '?' bilinmeyen kategori olarak -1'e kodlandı"
        assert encoded == expected_code, f"'{column}': imputer devreye girmedi"


def test_missing_field_falls_back_to_metadata_default(bundle: ModelBundle) -> None:
    """K3: verilmeyen alan `metadata.defaults` değeriyle dolar."""
    frame = bundle.prepare_row({})
    for column in bundle.input_order:
        assert frame[column].iloc[0] == bundle.defaults[column]


def test_prepare_row_applies_order_columns_step(bundle: ModelBundle) -> None:
    """`order_columns` adımı: kolon adları ve sırası sözleşmeyle birebir."""
    # Anahtarlar kasten ters sırada verildi.
    payload = {column: bundle.defaults[column] for column in reversed(bundle.input_order)}
    frame = bundle.prepare_row(payload)
    assert list(frame.columns) == bundle.input_order


def test_explicit_null_is_treated_as_missing(client: TestClient) -> None:
    """`null` göndermek alanı hiç göndermemekle aynı sonucu verir."""
    explicit = client.post("/predict", json={"witnesses": None, "age": None}).json()
    omitted = client.post("/predict", json={}).json()
    assert explicit == omitted


# --------------------------------------------------------------------------- #
# 7) SHAP çıktısı (K7)
# --------------------------------------------------------------------------- #


def test_shap_output_shape_and_naming(client: TestClient, bundle: ModelBundle) -> None:
    """34 eleman, abs(value) azalan sıralı, ham önek yok, alanlar sözleşmeye uygun."""
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    shap_values = body["shap_values"]

    assert len(shap_values) == 34
    assert len(shap_values) == len(bundle.input_order)

    for item in shap_values:
        assert set(item) == {"feature", "value", "base_value"}
        assert not item["feature"].startswith("cat__")
        assert not item["feature"].startswith("remainder__")
        assert math.isfinite(item["value"])

    # Tüm feature'lar tam olarak bir kez dönüyor.
    assert sorted(item["feature"] for item in shap_values) == sorted(bundle.input_order)

    magnitudes = [abs(item["value"]) for item in shap_values]
    assert magnitudes == sorted(magnitudes, reverse=True), "abs(value) azalan sırada değil"

    base_values = {item["base_value"] for item in shap_values}
    assert len(base_values) == 1, "base_value tüm elemanlarda aynı olmalı"


def test_shap_additivity_proves_correct_class_axis(client: TestClient) -> None:
    """SHAP'in temel özdeşliği: sum(value) + base_value = ham skor (log-odds).

    Bu testin asıl işi ŞEKİL VARSAYIMINI kanıtlamak: yanlış sınıf ekseni ya da
    yanlış satır seçilseydi eşitlik bozulurdu. `shap 0.52 + lightgbm 4.6`
    ikili modelde (1, 34) ndarray döndürüyor, ama sürüm değişip sınıf başına
    liste dönmeye başlarsa bu test kırmızıya döner.
    """
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()

    shap_values = body["shap_values"]
    margin = sum(item["value"] for item in shap_values) + shap_values[0]["base_value"]
    reconstructed = 1.0 / (1.0 + math.exp(-margin))

    assert reconstructed == pytest.approx(body["fraud_probability"], abs=1e-9)


def test_shap_values_stay_aligned_with_the_booster_contributions(bundle: ModelBundle) -> None:
    """SHAP katkıları LightGBM booster'ının `pred_contrib` çıktısıyla POZİSYON
    bazında hizalı kalmalı.

    NEDEN BU TEST VAR (reviewer MINOR-1'in cevabı):
    Eski "yön testi" totolojikti — matematiksel olarak toplanabilirlikten
    türüyordu ve asla kırmızıya dönemezdi. Daha da önemlisi, toplanabilirlik
    feature adlarının doğru eşlendiğini KANITLAMAZ: SHAP değerlerini kendi
    aralarında karıştırsanız bile toplam aynı kalır, yani additivity testi
    geçmeye devam eder.

    REFERANSIN SINIRI — BU ORACLE BAĞIMSIZ DEĞİLDİR:
    `shap.TreeExplainer` bu konfigürasyonda (LightGBM + objective='binary')
    hesabı kendisi yapmaz; `shap/explainers/_tree.py` içinde
    `original_model.predict(X, pred_contrib=True)` çağırarak DOĞRUDAN booster'a
    delege eder. Yani karşılaştırdığımız iki taraf aynı sayısal kaynaktan
    besleniyor ve bu test SHAP'in matematiğini DOĞRULAMAZ — böyle bir iddiada
    bulunmamalı.

    O hâlde neyi doğruluyor: kendi kodumuzun eşleme katmanını. `values` dizisini
    `transformed_display_names` ile eşleştiren `zip`, `abs`'e göre sıralama ve
    pozitif sınıf ekseni seçimi — bunların hepsi katkıları adlardan koparabilir
    ve sonuç yine "geçerli" görünür. Öldürdüğü mutasyonlar: adların
    kaydırılması/karıştırılması, `zip` off-by-one, yanlış satır seçimi,
    sıralamanın değerleri adlardan koparması.
    """
    payload = {
        "incident_severity": "Major Damage",
        "insured_hobbies": "chess",
        "auto_year": 2010,
        "witnesses": 2,
        "capital-gains": 30000,
        "total_claim_amount": 55000,
    }

    booster = bundle.pipeline.named_steps["model"].booster_
    transformed = bundle._transform(bundle.prepare_row(payload))
    contributions = booster.predict(transformed, pred_contrib=True)

    assert contributions.shape == (1, len(bundle.display_names) + 1)

    result = bundle.predict(payload)
    by_feature = {item["feature"]: item["value"] for item in result["shap_values"]}

    # Bizim temiz adlarımız booster'ın ham adlarıyla POZİSYON bazında eşleşmeli.
    for position, display_name in enumerate(bundle.display_names):
        assert by_feature[display_name] == contributions[0, position], (
            f"'{display_name}' (pozisyon {position}) booster katkısıyla uyuşmuyor"
        )

    base_value = result["shap_values"][0]["base_value"]
    assert base_value == contributions[0, -1]


def test_shap_feature_names_follow_the_transformed_column_order(bundle: ModelBundle) -> None:
    """Temiz adlar, booster'ın ham feature adlarının önek atılmış hâli olmalı.

    Yukarıdaki oracle testi değerleri bağlar; bu da adlandırmanın kaynağını
    bağlar. İkisi birlikte "i'inci katkı, i'inci feature'a aittir" iddiasını
    her iki uçtan kapatır.
    """
    raw_names = list(bundle.pipeline.named_steps["model"].booster_.feature_name())
    assert [name.split("__", 1)[-1] for name in raw_names] == bundle.display_names


# --------------------------------------------------------------------------- #
# 8) PII sızıntısı — /predict
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("payload", [PREDICT_REQUEST_EXAMPLE, {}])
def test_predict_never_leaks_pii_column_names(client: TestClient, payload: dict) -> None:
    """`/predict` yanıtının HAM METNİNDE PII kolon adı geçmez.

    `/predict` yanıtı girdiyi yankılamadığı için PII *değeri* zaten dönemez;
    bu test şema genişletilirse (ör. "requestin echo'su" eklenirse) alarm verir.
    """
    raw = client.post("/predict", json=payload).text
    for column in PII_COLUMNS:
        assert column not in raw
    # Atılan diğer kolonlar da yanıt yüzeyinde işi yok.
    assert "auto_model" not in raw
    assert "incident_date" not in raw


# --------------------------------------------------------------------------- #
# 9) Şema <-> artefakt senkronu
# --------------------------------------------------------------------------- #


def test_request_schema_covers_every_pipeline_input(bundle: ModelBundle) -> None:
    """Pydantic modeli, pipeline'ın 34 girdisinin hepsini kapsıyor mu?

    Eksik alan, o alanın API'den HİÇ verilememesi (hep default kalması)
    demektir — sessiz bir işlevsellik kaybı.
    """
    aliases = {
        field.alias or name for name, field in PredictRequest.model_fields.items()
    }
    assert aliases == set(bundle.input_order)


def test_literal_choices_match_metadata(metadata: dict[str, Any]) -> None:
    """Her kategorik alanın `Literal` seçenekleri eğitim kategorileriyle aynı.

    Kategori listeleri `schemas.py` içinde okunabilirlik için elle yazıldı;
    bu test onları artefakta bağlar. Model yeniden eğitilip bir kategori
    eklenirse/çıkarsa test kırmızıya döner — şema sessizce eskimez.
    """
    training_ranges = metadata["training_ranges"]
    question_mark_columns = set(
        next(
            step
            for step in metadata["preprocessing_contract"]["caller_must_apply_before_predict"]
            if step["step"] == "question_mark_to_nan"
        )["columns"]
    )

    categorical = metadata["feature_list"]["categorical_features"]
    for column in categorical:
        field = PredictRequest.model_fields[column]
        # Annotation: Literal[...] | None -> önce Optional'ı aç.
        literal_args = set()
        for member in get_args(field.annotation):
            literal_args.update(get_args(member))

        expected = set(training_ranges[column]["categories"])
        if column in question_mark_columns:
            expected.add("?")

        assert literal_args == expected, f"'{column}' Literal seçenekleri metadata ile uyuşmuyor"


def test_numeric_bounds_are_supersets_of_training_ranges(metadata: dict[str, Any]) -> None:
    """K5: Pydantic sınırı eğitim aralığını KAPSAMALI.

    İki yönlü kontrol:
      * Eğitimdeki bir değer 422 almamalı (sınır üst küme olmalı).
      * `incident_hour_of_the_day` dışında sınır eğitim aralığına EŞİT
        olmamalı; eşit olsaydı Faz 2 guardrail'i o alanda hiç tetiklenemezdi.
    """
    # Fiziksel sınırı gerçekten eğitim aralığıyla çakışan tek alan.
    naturally_bounded = {"incident_hour_of_the_day"}

    # Sınırları YAYINLANAN JSON Schema'dan okuyoruz, Pydantic'in iç
    # alanlarından değil: frontend geliştiricisinin OpenAPI'de gördüğü
    # sözleşmenin ta kendisi bu. `int | None` alanlarda kısıtlar
    # anyOf'un null olmayan dalında durur.
    properties = PredictRequest.model_json_schema()["properties"]

    for column in metadata["feature_list"]["numeric_features"]:
        branches = [
            branch
            for branch in properties[column].get("anyOf", [properties[column]])
            if branch.get("type") != "null"
        ]
        assert len(branches) == 1, f"'{column}' için beklenmeyen şema dalı"
        branch = branches[0]

        assert "minimum" in branch and "maximum" in branch, f"'{column}' için ge/le tanımlı değil"
        low, high = float(branch["minimum"]), float(branch["maximum"])

        train_min = float(metadata["training_ranges"][column]["min"])
        train_max = float(metadata["training_ranges"][column]["max"])

        assert low <= train_min, f"'{column}' alt sınırı eğitim minimumunu kesiyor"
        assert high >= train_max, f"'{column}' üst sınırı eğitim maksimumunu kesiyor"

        if column not in naturally_bounded:
            assert (low, high) != (train_min, train_max), (
                f"'{column}' Pydantic sınırı eğitim aralığına eşit — guardrail tetiklenemez"
            )


def test_training_extremes_are_accepted(client: TestClient, metadata: dict[str, Any]) -> None:
    """Eğitim setinin min/max değerleri API'den geçebilmeli.

    Bir üstteki test sınırları statik karşılaştırıyor; bu test aynı iddiayı
    gerçek HTTP isteğiyle kanıtlıyor.
    """
    for column in metadata["feature_list"]["numeric_features"]:
        ranges = metadata["training_ranges"][column]
        for bound in ("min", "max"):
            response = client.post("/predict", json={column: ranges[bound]})
            assert response.status_code == 200, (
                f"eğitimdeki {column}={ranges[bound]} reddedildi: {response.text}"
            )


# --------------------------------------------------------------------------- #
# 10) risk_level eşikleri (K6)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (RISK_THRESHOLD_MEDIUM - 1e-9, "low"),
        (RISK_THRESHOLD_MEDIUM, "medium"),
        (0.5, "medium"),
        (RISK_THRESHOLD_HIGH - 1e-9, "medium"),
        (RISK_THRESHOLD_HIGH, "high"),
        (1.0, "high"),
    ],
)
def test_risk_level_thresholds(probability: float, expected: str) -> None:
    """Eşikler kapalı-alt/açık-üst: low < 0.35 <= medium < 0.65 <= high."""
    assert classify_risk(probability) == expected


# --------------------------------------------------------------------------- #
# 11) CORS (K10)
# --------------------------------------------------------------------------- #


def test_cors_env_var_is_parsed_as_comma_separated_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://a.netlify.app, https://b.example , ")
    assert get_allowed_origins() == ["https://a.netlify.app", "https://b.example"]


def test_cors_default_is_the_vite_dev_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Değişken hiç SET EDİLMEMİŞSE varsayılan devreye girer (hata değil)."""
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    assert get_allowed_origins() == ["http://localhost:5173"]
    assert DEFAULT_ALLOWED_ORIGINS == "http://localhost:5173"


@pytest.mark.parametrize(
    "value",
    ["*", "http://localhost:5173,*", " * ", "https://*.netlify.app"],
)
def test_cors_wildcard_is_rejected_at_startup(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """K10 artık ortam değişkeni yolunda da zorlanıyor (reviewer MAJOR-2).

    Eskiden `ALLOWED_ORIGINS="*"` gerçek bir wildcard üretiyordu: kural sadece
    varsayılan değerde geçerliydi, tek bir ortam değişkeniyle deliniyordu.
    """
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, value)
    with pytest.raises(CorsConfigurationError, match="wildcard"):
        get_allowed_origins()
    # Uygulama da açılmamalı — hata yapılandırma katmanında kalmıyor.
    with pytest.raises(CorsConfigurationError):
        create_app()


@pytest.mark.parametrize("value", ["", "   ", ",", " , , "])
def test_cors_explicitly_empty_is_rejected_at_startup(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Boş `ALLOWED_ORIGINS` sessizce tüm istemcileri kapatmasın (MINOR-5)."""
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, value)
    with pytest.raises(CorsConfigurationError, match="geçerli origin"):
        get_allowed_origins()


def test_cors_wiring_reflects_allowed_origin_and_rejects_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORS KABLOLAMASININ kendisi test ediliyor (reviewer MAJOR-1).

    `create_app()` fabrikası origin'leri çağrıldığı anda okuduğu için testler
    ortamı değiştirip yeni bir uygulama kurabiliyor. Eskiden origin listesi
    import anında dondurulduğundan bu yol test edilemiyordu — ve test
    edilemeyen yolu reviewer'ın M21/M23/M24 mutasyonları sessizce geçmişti.
    """
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://izinli.example,https://ikinci.example")

    with TestClient(create_app()) as scoped:
        allowed = scoped.get("/health", headers={"Origin": "https://izinli.example"})
        assert allowed.headers.get("access-control-allow-origin") == "https://izinli.example"

        second = scoped.get("/health", headers={"Origin": "https://ikinci.example"})
        assert second.headers.get("access-control-allow-origin") == "https://ikinci.example"

        # İzinsiz origin yansıtılmamalı (M21: origin'i sabit kodlayan mutasyon).
        denied = scoped.get("/health", headers={"Origin": "https://kotu.example"})
        assert denied.headers.get("access-control-allow-origin") is None

        # Varsayılan origin artık listede olmadığı için o da yansıtılmamalı.
        stale = scoped.get("/health", headers={"Origin": "http://localhost:5173"})
        assert stale.headers.get("access-control-allow-origin") is None


def test_cors_preflight_limits_methods_and_disables_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight yanıtı yalnızca GET/POST'a izin verir, credentials kapalıdır.

    M23 (`allow_methods=["*"]`) ve M24 (`allow_credentials=True`) mutasyonlarını
    öldüren test budur.
    """
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://izinli.example")

    with TestClient(create_app()) as scoped:
        preflight = scoped.options(
            "/predict",
            headers={
                "Origin": "https://izinli.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 200

        methods = {
            m.strip().upper()
            for m in preflight.headers.get("access-control-allow-methods", "").split(",")
            if m.strip()
        }
        assert methods == {"GET", "POST"}, f"beklenmeyen metod kümesi: {methods}"
        assert "*" not in preflight.headers.get("access-control-allow-methods", "")

        # Credentials KAPALI olmalı: açık olsaydı başlık "true" olarak dönerdi.
        assert preflight.headers.get("access-control-allow-credentials") is None

        # Yasaklı bir metodun preflight'ı ONAYLANMAMALI.
        # Starlette bu durumda 400 + "Disallowed CORS method" döner; origin
        # başlığı yanıtta olsa da tarayıcı isteği yine de engeller, belirleyici
        # olan durum kodudur.
        rejected = scoped.options(
            "/predict",
            headers={
                "Origin": "https://izinli.example",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        assert rejected.status_code == 400
        assert "Disallowed CORS method" in rejected.text


def test_cors_headers_are_present_on_error_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """413/422 gibi hata yanıtları da CORS başlıklarını almalı.

    Aksi hâlde tarayıcı istemcisi hatayı okuyamaz, yalnızca opak bir ağ hatası
    görür — middleware sırasının (CORS en dışta) somut gerekçesi budur.
    """
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://izinli.example")

    with TestClient(create_app()) as scoped:
        invalid = scoped.post(
            "/predict",
            json={"witnesses": 999},
            headers={"Origin": "https://izinli.example"},
        )
        assert invalid.status_code == 422
        assert invalid.headers.get("access-control-allow-origin") == "https://izinli.example"

        oversized = scoped.post(
            "/predict",
            content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
            headers={"Origin": "https://izinli.example", "Content-Type": "application/json"},
        )
        assert oversized.status_code == 413
        assert oversized.headers.get("access-control-allow-origin") == "https://izinli.example"


# --------------------------------------------------------------------------- #
# 12) 422 gövdesi istemci verisini geri yansıtmamalı (reviewer MINOR-2)
# --------------------------------------------------------------------------- #


def test_validation_error_does_not_echo_submitted_values(client: TestClient) -> None:
    """Hatalı istekte gönderilen DEĞER yanıtta geçmemeli.

    Pydantic'in varsayılan gövdesi `input` alanında ham değeri geri yollar.
    Bu API'ye giden gövde sigorta talebi verisidir; hatalı bir istek onu
    log'lara, ters proxy'lere ve tarayıcı konsoluna taşırdı.
    """
    canary = "GIZLI-TALEP-VERISI-7f3a91"
    response = client.post(
        "/predict",
        json={"insured_hobbies": canary, "witnesses": 4242, "policy_annual_premium": -12345.6},
    )

    assert response.status_code == 422
    raw = response.text
    assert canary not in raw, "gönderilen değer yanıtta yankılandı"
    assert "4242" not in raw
    assert "12345.6" not in raw

    # `input` ve `ctx` alanları tamamen düşmüş olmalı.
    for error in response.json()["detail"]:
        assert set(error) == {"loc", "msg", "type"}


def test_validation_error_still_tells_which_field_failed(client: TestClient) -> None:
    """Sanitizasyon teşhis değerini BOZMAMALI — frontend alan bazlı hata gösterir."""
    response = client.post("/predict", json={"witnesses": 999, "age": 5})
    assert response.status_code == 422

    detail = response.json()["detail"]
    failed_fields = {error["loc"][-1] for error in detail}
    assert failed_fields == {"witnesses", "age"}

    for error in detail:
        assert error["loc"][0] == "body"
        assert error["msg"], "hata mesajı boş kalmamalı"
        assert error["type"], "makine-okunur hata tipi kaybolmamalı"


def test_unknown_field_error_names_the_field_but_not_its_value(client: TestClient) -> None:
    """Bilinmeyen alanda ANAHTAR adı döner, DEĞER dönmez."""
    response = client.post("/predict", json={"gizli_alan": "cok-gizli-deger-991"})
    assert response.status_code == 422
    assert "gizli_alan" in response.text
    assert "cok-gizli-deger-991" not in response.text


# --------------------------------------------------------------------------- #
# 13) Gövde boyut sınırı (reviewer MINOR-6)
# --------------------------------------------------------------------------- #


def test_body_within_limit_is_processed_normally(client: TestClient) -> None:
    """Sınırın altındaki gövde normal akışa girer (200 ya da 422)."""
    response = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE)
    assert response.status_code == 200


def test_oversized_body_is_rejected_with_413(client: TestClient) -> None:
    """`Content-Length` ile bildirilen büyük gövde okunmadan reddedilir."""
    payload = b'{"insured_hobbies":"' + b"a" * (MAX_REQUEST_BODY_BYTES + 1) + b'"}'
    response = client.post(
        "/predict", content=payload, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert "detail" in response.json()


def test_oversized_chunked_body_is_rejected_with_413(client: TestClient) -> None:
    """`Content-Length` YOKKEN de korunmalıyız (chunked transfer encoding).

    Sadece başlığa güvenmek, başlığı hiç göndermeyen bir istemciye sınırsız
    gövde hakkı tanımak olurdu.
    """

    def chunks() -> Any:
        yield b'{"insured_hobbies":"'
        for _ in range(16):
            yield b"a" * 8192
        yield b'"}'

    response = client.post(
        "/predict", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_body_just_under_the_limit_reaches_validation(client: TestClient) -> None:
    """Sınırın hemen altındaki gövde 413 DEĞİL, normal doğrulama hatası alır.

    Sınırın yanlış tarafa kaymadığını (off-by-one) kanıtlar.
    """
    filler = "a" * (MAX_REQUEST_BODY_BYTES - 200)
    response = client.post(
        "/predict", content=json.dumps({"insured_hobbies": filler}).encode()
    )
    assert response.status_code == 422, response.status_code


# --------------------------------------------------------------------------- #
# 14) OpenAPI yüzeyi (reviewer MINOR-3)
# --------------------------------------------------------------------------- #


def test_openapi_does_not_leak_internal_or_pii_names(client: TestClient) -> None:
    """`/openapi.json` de public bir yüzeydir ve taranmalıdır.

    Yasaklı adlar `/model-info` yanıtından çıkarılmıştı ama class docstring'leri
    üzerinden OpenAPI `description` alanlarına sızıyordu. Açıklamalar `#`
    yorumlarına taşındı; bu test yüzeyin temiz kalmasını garanti eder.
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200

    raw = response.text
    for key in (*FORBIDDEN_MODEL_INFO_KEYS, "n_estimators", "learning_rate", *PII_COLUMNS):
        assert key not in raw, f"'{key}' /openapi.json içine sızmış"


def test_openapi_still_documents_the_public_contract(client: TestClient) -> None:
    """Sızıntı temizliği dokümantasyonu boşaltmamalı."""
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == {"/health", "/predict", "/model-info"}
    assert "capital-gains" in spec["components"]["schemas"]["PredictRequest"]["properties"]


# --------------------------------------------------------------------------- #
# 15) Eş zamanlılık ve tek-dönüşüm kısayolu (reviewer MINOR-4 + regresyon)
# --------------------------------------------------------------------------- #


def test_concurrent_predictions_are_bitwise_identical(client: TestClient) -> None:
    """50 paralel `/predict` aynı girdi için bit bazında aynı sonucu vermeli.

    Paylaşılan durum (`explainer.expected_value` her çağrıda yeniden atanıyor)
    kilitle korunuyor; bu test o korumanın regresyonunu yakalar.
    """
    workers = 50

    def call() -> tuple[float, str]:
        body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
        return body["fraud_probability"], json.dumps(body["shap_values"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: call(), range(workers)))

    assert len({probability for probability, _ in results}) == 1
    assert len({shap_blob for _, shap_blob in results}) == 1


def test_split_transform_matches_full_pipeline_bitwise(bundle: ModelBundle) -> None:
    """`predict()` kısayolu tam `pipeline.predict_proba` ile aynı sonucu verir.

    `predict()` performans için preprocessor'ı bir kez çalıştırıp aynı matrisi
    hem modele hem explainer'a veriyor. Bu test kısayolun sessizce sapmadığını
    kanıtlar (açılışta da `_verify_transform_equivalence` ile kontrol edilir).
    """
    payload = PredictRequest(**PREDICT_REQUEST_EXAMPLE).model_dump(by_alias=True)
    frame = bundle.prepare_row(payload)

    via_pipeline = bundle.pipeline.predict_proba(frame)
    via_steps = bundle.pipeline.named_steps["model"].predict_proba(bundle._transform(frame))

    assert np.array_equal(via_pipeline, via_steps)
    assert bundle.predict(payload)["fraud_probability"] == float(via_pipeline[0, 1])


# --------------------------------------------------------------------------- #
# 16) Kütüphane sürüm sapması (reviewer C1)
# --------------------------------------------------------------------------- #


def test_version_drift_emits_a_warning(
    bundle: ModelBundle, caplog: pytest.LogCaptureFixture
) -> None:
    """Eğitim ve çalışma zamanı sürümleri ayrışırsa UYARI basılır.

    Patlatmıyoruz: sürüm sapması çoğu zaman zararsızdır ve her sapmada açılışı
    reddetmek Faz 7'de deploy'u gereksiz yere kilitlerdi. Ama sessiz kalmak,
    "model neden farklı skor veriyor?" sorusunun en pahalı hâlidir.
    """
    drifted = copy.deepcopy(bundle.metadata)
    drifted["library_versions"]["lightgbm"] = "0.0.1-eski"
    drifted["library_versions"]["scikit-learn"] = "0.0.2-eski"

    probe = ModelBundle(bundle.pipeline, drifted)
    with caplog.at_level(logging.WARNING, logger="app.model"):
        probe._warn_on_library_version_drift()

    assert "sürüm sapması" in caplog.text
    assert "lightgbm" in caplog.text
    assert "scikit-learn" in caplog.text


def test_no_warning_when_versions_match(
    bundle: ModelBundle, caplog: pytest.LogCaptureFixture
) -> None:
    """Sürümler uyuşuyorsa log kirletilmez."""
    with caplog.at_level(logging.WARNING, logger="app.model"):
        bundle._warn_on_library_version_drift()
    assert "sürüm sapması" not in caplog.text


def test_version_drift_is_not_exposed_on_health(client: TestClient) -> None:
    """Sapma bilgisi API sözleşmesine sızmaz — operatör sinyali, istemci verisi değil."""
    body = client.get("/health").json()
    assert set(body) == {"status", "model_loaded", "model_version"}


# --------------------------------------------------------------------------- #
# 17) Feature etki verisi (F9) — "işaretle ve göster" kararının API tarafı
# --------------------------------------------------------------------------- #


def test_feature_influence_is_in_metadata_and_covers_every_field(
    metadata: dict[str, Any],
) -> None:
    """34 feature'ın hepsi için ölçüm var ve anahtarlar API alan adlarıyla birebir.

    Anahtarların API adlarıyla eşleşmesi şart: frontend ayrı bir eşleme tablosu
    tutmak zorunda kalırsa o tablo model değişince sessizce eskir.
    """
    influence = metadata["feature_influence"]
    features = influence["features"]

    assert len(features) == 34
    assert set(features) == set(metadata["feature_list"]["pipeline_input_order"])

    # Tireli alan adı korunmalı (Pydantic alias'ı ile aynı).
    assert "capital-gains" in features
    assert "capital_gains" not in features

    for name, entry in features.items():
        assert set(entry) == {"split_count", "gain", "has_influence"}
        assert entry["has_influence"] == (entry["split_count"] > 0), name
        assert entry["split_count"] >= 0
        assert entry["gain"] >= 0.0


def test_feature_influence_summary_matches_the_per_feature_data(
    metadata: dict[str, Any],
) -> None:
    """Özet sayılar detayla tutarlı — özet elle yazılmış bir sabit değil."""
    influence = metadata["feature_influence"]
    features = influence["features"]
    summary = influence["summary"]

    dead = sorted(name for name, entry in features.items() if not entry["has_influence"])

    assert summary["n_features"] == len(features)
    assert summary["n_without_influence"] == len(dead)
    assert summary["n_with_influence"] == len(features) - len(dead)
    assert summary["features_without_influence"] == dead
    assert summary["n_with_influence"] + summary["n_without_influence"] == summary["n_features"]


def test_sixteen_features_are_dead_and_insured_sex_is_one_of_them(
    metadata: dict[str, Any],
) -> None:
    """Ölçülen gerçek: 16 feature ölü, `insured_sex` bunlardan biri."""
    summary = metadata["feature_influence"]["summary"]

    assert summary["n_without_influence"] == 16
    assert summary["n_with_influence"] == 18
    assert "insured_sex" in summary["features_without_influence"]
    # Fairness beyanını doğrudan ilgilendiren diğer ölüler.
    for feature in ("insured_relationship", "policy_state", "incident_state"):
        assert feature in summary["features_without_influence"]
    # Gerçekten kullanılanlar ölü listesinde OLMAMALI.
    for feature in ("insured_hobbies", "incident_severity", "age", "insured_education_level"):
        assert feature not in summary["features_without_influence"]


def test_feature_influence_matches_the_booster_exactly(
    bundle: ModelBundle, metadata: dict[str, Any]
) -> None:
    """Metadata'daki sayılar booster'dan hesaplananla BİREBİR aynı.

    Elle yazılmış sabit olmadığının kanıtı: aynı sayıyı bağımsız olarak
    booster'dan okuyup karşılaştırıyoruz.
    """
    booster = bundle.pipeline.named_steps["model"].booster_
    split_counts = booster.feature_importance(importance_type="split")
    gains = booster.feature_importance(importance_type="gain")
    features = metadata["feature_influence"]["features"]

    for position, display_name in enumerate(bundle.display_names):
        entry = features[display_name]
        assert entry["split_count"] == int(split_counts[position]), display_name
        assert entry["gain"] == pytest.approx(float(gains[position])), display_name
        assert entry["has_influence"] == (int(split_counts[position]) > 0)


def test_model_info_exposes_feature_influence(client: TestClient) -> None:
    """`/model-info` bölümü döndürüyor ve beyaz liste bozulmamış."""
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()

    influence = body["feature_influence"]
    assert len(influence["features"]) == 34
    assert influence["summary"]["n_without_influence"] == 16
    assert influence["features"]["insured_sex"]["has_influence"] is False
    assert influence["features"]["insured_hobbies"]["has_influence"] is True
    assert "capital-gains" in influence["features"]

    # Beyaz liste hâlâ sızdırmıyor.
    raw = response.text
    for key in FORBIDDEN_MODEL_INFO_KEYS:
        assert key not in raw


def test_dead_features_have_exactly_zero_shap_value(client: TestClient) -> None:
    """`has_influence: false` olan her feature'ın SHAP katkısı TAM 0.0.

    `feature_influence` sözleşmesinin uçtan uca doğrulaması: frontend "bu alan
    modeli etkilemiyor" rozetini bu veriye dayanarak basacak, dolayısıyla
    rozetin gerçekle uyuştuğunu kanıtlamak zorundayız.
    """
    influence = client.get("/model-info").json()["feature_influence"]["features"]
    shap_values = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()["shap_values"]
    by_feature = {item["feature"]: item["value"] for item in shap_values}

    dead = [name for name, entry in influence.items() if not entry["has_influence"]]
    assert len(dead) == 16

    for name in dead:
        assert by_feature[name] == 0.0, f"'{name}' ölü ama SHAP katkısı sıfır değil"

    # Ters yön: etkili feature'ların hepsi sıfır olsaydı test anlamsız olurdu.
    alive = [name for name, entry in influence.items() if entry["has_influence"]]
    assert any(by_feature[name] != 0.0 for name in alive)


# --------------------------------------------------------------------------- #
# 18) Fairness: beyan ile ölçüm ayrıştırıldı (F10)
# --------------------------------------------------------------------------- #


def test_fairness_entries_carry_both_declaration_and_measurement(
    client: TestClient,
) -> None:
    """Her korunan/vekil nitelikte hem beyan hem ölçüm alanı var."""
    fairness = client.get("/model-info").json()["fairness"]
    entries = (
        fairness["protected_attributes_used_as_features"] + fairness["proxy_risk_attributes"]
    )
    assert entries

    for entry in entries:
        assert entry["used_as_model_feature"] is True, "beyan korunmalı"
        assert isinstance(entry["split_count"], int)
        assert entry["has_influence"] == (entry["split_count"] > 0)


def test_fairness_split_counts_come_from_the_booster(
    client: TestClient, bundle: ModelBundle
) -> None:
    """Fairness bölümündeki `split_count` booster'dan hesaplananla birebir aynı.

    Elle yazılmış sabit OLMADIĞININ kanıtı. Mutasyonla da doğrulandı: sayıyı
    sabitleyen bir değişiklik bu testi kırmızıya döndürür.
    """
    booster = bundle.pipeline.named_steps["model"].booster_
    split_counts = booster.feature_importance(importance_type="split")
    expected = {
        name: int(split_counts[position])
        for position, name in enumerate(bundle.display_names)
    }

    fairness = client.get("/model-info").json()["fairness"]
    entries = (
        fairness["protected_attributes_used_as_features"] + fairness["proxy_risk_attributes"]
    )

    for entry in entries:
        assert entry["split_count"] == expected[entry["feature"]], entry["feature"]


def test_fairness_reports_measured_zero_influence_for_sex(client: TestClient) -> None:
    """`insured_sex` beyan olarak verilmiş ama ölçülen etkisi sıfır."""
    fairness = client.get("/model-info").json()["fairness"]
    by_feature = {
        entry["feature"]: entry for entry in fairness["protected_attributes_used_as_features"]
    }

    sex = by_feature["insured_sex"]
    assert sex["used_as_model_feature"] is True
    assert sex["split_count"] == 0
    assert sex["has_influence"] is False

    # Yaş ise gerçekten kullanılıyor — ölçüm ayrımı anlamlı çalışıyor.
    assert by_feature["age"]["has_influence"] is True


def test_fairness_does_not_claim_the_model_is_fair(client: TestClient) -> None:
    """Ölçüm eklendi diye "model adil" sonucuna ATLANMAMALI.

    `has_influence: false` yalnızca "bu eğitilmiş ağaç kümesinde ölçülen etki
    sıfır" demektir. Denetim hâlâ yapılmadı ve vekil feature'lar sinyali geri
    taşıyabilir; model card bunu açıkça söylemek zorunda.
    """
    fairness = client.get("/model-info").json()["fairness"]

    assert fairness["audit_performed"] is False
    assert fairness["status"] == "declared_not_audited"
    assert fairness["audit_metrics_computed"] == []
    assert fairness["production_requirements"], "üretim gereklilikleri silinmemeli"

    text = _fold(fairness["notes"] + " " + fairness["field_semantics"])
    # Ölçümün sınırları açıkça yazılmış olmalı.
    assert "denetim" in text
    assert "vekil" in text
    assert "anlamina gelmez" in text or "yerine gecmez" in text

    # Kesin/aklayıcı bir iddia geçmemeli.
    for forbidden in ("model adil", "ayrimcilik yok", "ayrimcilik yapmiyor", "bias yok"):
        assert forbidden not in text, f"aşırı iddia: {forbidden!r}"


# --------------------------------------------------------------------------- #
# 12) Doğrudan model katmanı
# --------------------------------------------------------------------------- #


def test_bundle_predict_matches_endpoint(client: TestClient, bundle: ModelBundle) -> None:
    """HTTP katmanı hesap yapmıyor: endpoint çıktısı = bundle çıktısı."""
    direct = bundle.predict(PredictRequest(**PREDICT_REQUEST_EXAMPLE).model_dump(by_alias=True))
    via_http = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert direct["fraud_probability"] == pytest.approx(via_http["fraud_probability"], abs=0)
    assert direct["risk_level"] == via_http["risk_level"]


def test_unknown_category_never_reaches_the_model(bundle: ModelBundle) -> None:
    """Encoder'ın -1 yolu API üzerinden ERİŞİLEMEZ olmalı.

    Pipeline `unknown_value=-1` ile kurulu, yani bilinmeyen kategori sessizce
    -1'e kodlanır. `Literal` bunu API seviyesinde kapattığı için tek bir
    kategorik alanda bile -1 görmemeliyiz.
    """
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    frame = bundle.prepare_row(PredictRequest(**PREDICT_REQUEST_EXAMPLE).model_dump(by_alias=True))
    transformed = preprocessor.transform(frame)

    categorical = [c for c in transformed.columns if c.startswith("cat__")]
    codes = transformed[categorical].to_numpy(dtype=float)
    assert not np.any(codes < 0), "geçerli bir istekte bilinmeyen kategori (-1) oluştu"


# --------------------------------------------------------------------------- #
# 19) Doğrulama turu — hayatta kalan mutasyonların kapatılması
# --------------------------------------------------------------------------- #
#
# Aşağıdaki dört test, önceki turda MUTASYONLA ÖLDÜRÜLEMEYEN iddiaları bağlar.
# Bir iddianın metinde yazılı olması onu doğru yapmaz; her biri ancak ilgili
# kodu bozunca kırmızıya dönen bir testle kanıtlanmış sayılır.


def test_shap_lock_is_actually_held_during_explanation(
    bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explainer çağrılırken `_shap_lock` GERÇEKTEN tutuluyor olmalı.

    NEDEN AYRI BİR TEST GEREKTİ (hayatta kalan mutasyon A-M8):
    `test_concurrent_predictions_are_bitwise_identical` kilidi kaldıran
    mutasyonu ÖLDÜREMİYOR ve bu şaşırtıcı değil — shap'in her çağrıda yeniden
    atadığı `expected_value` hep AYNI değere ayarlanıyor, dolayısıyla yarış
    durumu çıktıda gözlemlenebilir bir fark üretmiyor. Yani kilit, bugün
    ölçülebilir bir bug'ı değil, kütüphanenin paylaşılan durumu mutasyona
    uğratma DAVRANIŞINI kapatıyor; korunan şey gelecekteki bir shap sürümünde
    o değerin girdiye bağlı hâle gelmesi.

    Sonuç bazlı bir test bunu asla kanıtlayamaz, o yüzden korumanın kendisini
    doğrudan gözlemliyoruz: açıklama üretilirken kilit tutuluyor mu?
    `with self._shap_lock:` satırı silinirse burası kırmızıya döner.
    """
    original_explainer = bundle.explainer
    lock_states: list[bool] = []

    class _LockObservingExplainer:
        """Explainer'ı sarmalar; çağrıldığı ANDA kilidin durumunu kaydeder."""

        def __call__(self, transformed: Any) -> Any:
            lock_states.append(bundle._shap_lock.locked())
            return original_explainer(transformed)

    monkeypatch.setattr(bundle, "explainer", _LockObservingExplainer())
    bundle.predict(PREDICT_REQUEST_EXAMPLE)

    assert lock_states == [True], (
        "SHAP açıklaması kilit tutulmadan üretildi — paylaşılan explainer "
        "durumu eş zamanlı isteklere açık."
    )


def test_load_runs_every_startup_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ModelBundle.load()` beş açılış doğrulamasının HEPSİNİ çağırmalı.

    NEDEN AYRI BİR TEST GEREKTİ (hayatta kalan mutasyon):
    `_verify_transform_equivalence()` ÇAĞRISINI `load()` içinden silen mutasyon
    tüm suite'i yeşil bırakıyordu. Metodun kendisi test ediliyordu, çağrıldığı
    ise değil — artefakt zaten tutarlı olduğu için doğrulamayı atlamak hiçbir
    gözlemlenebilir fark yaratmıyor. Ama koruma tam da tutarsız bir artefakt
    içindir: sessizce kaldırılırsa fail-fast garantisi kâğıt üstünde kalır.
    """
    verifications = (
        "_verify_contract",
        "_verify_feature_alignment",
        "_verify_transform_equivalence",
        "_verify_shap_additivity",
        "_warn_on_library_version_drift",
    )
    called: list[str] = []

    def spy(name: str) -> Any:
        original = getattr(ModelBundle, name)

        def wrapper(self: ModelBundle) -> Any:
            called.append(name)
            return original(self)

        return wrapper

    for name in verifications:
        monkeypatch.setattr(ModelBundle, name, spy(name))

    ModelBundle.load()

    assert set(called) == set(verifications), (
        f"açılışta atlanan doğrulama(lar): {sorted(set(verifications) - set(called))}"
    )
    # Sözleşme kontrolü ÖNCE gelmeli: diğer doğrulamalar metadata'nın
    # beklenen şekilde olduğunu varsayarak çalışıyor.
    assert called[0] == "_verify_contract"


def test_dead_features_change_neither_the_score_nor_their_own_shap_value(
    bundle: ModelBundle, metadata: dict[str, Any]
) -> None:
    """Ölü feature'ın DEĞERİ değişince ne olasılık ne de SHAP katkısı değişir.

    NEDEN AYRI BİR TEST GEREKTİ (fairness metninin aşırı-iddia denetimi):
    `fairness.field_semantics` şunu iddia ediyor — "split_count=0 olan bir
    feature bu eğitilmiş modelde hiçbir tahmini etkilemez ve SHAP katkısı HER
    ZAMAN tam 0.0'dır". Bu metin `/model-info` üzerinden sigorta müşterisine
    gösterilecek. Mevcut `test_dead_features_have_exactly_zero_shap_value` bunu
    TEK bir girdi üzerinde ölçüyordu; "her zaman" iddiası tek örnekle
    kanıtlanmaz.

    Burada her ölü feature'ın eğitimde görülen DEĞER UÇLARINI tek tek deneyip
    iki şeyi birden kanıtlıyoruz: (1) skor bit bazında sabit kalıyor,
    (2) o feature'ın kendi SHAP katkısı tam 0.0. `insured_sex` ve
    `insured_relationship` de bu kümede — yani "cinsiyeti değiştirmek skoru
    değiştirmiyor" artık bir beyan değil, ölçüm.

    DİKKAT — BU BİR FAIRNESS DENETİMİ DEĞİLDİR. Kanıtlanan tek şey, BU eğitilmiş
    ağaç kümesinde bu kolonların doğrudan etkisinin sıfır olduğudur. Vekil
    feature'lar (meslek, hobi, coğrafya) aynı sinyali dolaylı taşıyabilir ve
    grup bazlı hiçbir metrik hesaplanmadı — `fairness.notes` bunu zaten söylüyor.
    """
    influence = metadata["feature_influence"]["features"]
    ranges = metadata["training_ranges"]

    baseline = bundle.predict({})
    baseline_probability = baseline["fraud_probability"]

    dead = [name for name, entry in influence.items() if not entry["has_influence"]]
    assert len(dead) == 16, "ölü feature sayısı değişmiş — metadata ile test ayrışıyor"

    checked = 0
    for name in dead:
        spec = ranges[name]
        if spec["type"] == "categorical":
            candidates: list[Any] = list(spec["categories"])
        else:
            # Sayısal alanlarda eğitim aralığının iki ucu: etkisizlik iddiası
            # uçlarda da geçerli olmalı.
            candidates = [spec["min"], spec["max"]]

        for value in candidates:
            result = bundle.predict({name: value})
            assert result["fraud_probability"] == baseline_probability, (
                f"'{name}' ölü ilan edilmiş ama değeri {value!r} yapılınca skor değişti"
            )
            contribution = next(
                item["value"] for item in result["shap_values"] if item["feature"] == name
            )
            assert contribution == 0.0, (
                f"'{name}' ölü ilan edilmiş ama {value!r} girdisinde SHAP katkısı "
                f"{contribution!r}"
            )
            checked += 1

    # Testin gerçekten iş yaptığının kanıtı: 16 feature için çok sayıda değer.
    assert checked > 40, f"beklenenden az kombinasyon denendi: {checked}"

    # Ters yön — canlı bir feature'ı değiştirmek skoru DEĞİŞTİRMELİ, yoksa
    # yukarıdaki eşitlikler modelin hiçbir şeye tepki vermemesinden ibaret olurdu.
    moved = bundle.predict({"insured_hobbies": "chess"})["fraud_probability"]
    assert moved != baseline_probability


def test_model_info_projection_drops_unknown_nested_keys(bundle: ModelBundle) -> None:
    """Beyaz liste İÇ İÇE anahtarlarda da sızdırmıyor.

    NEDEN AYRI BİR TEST GEREKTİ (feature_influence sızıntı testi):
    Mevcut sızıntı testleri BUGÜN metadata'da var olan üç anahtarı
    (`preprocessing_contract`, `model_params`, `source_file`) arıyor. Ama beyaz
    listenin asıl vaadi geleceğe dönük: "yarın train_pipeline.py metadata'ya
    yeni bir alan eklerse o alan açıkça beyaz listeye alınana kadar dışarı
    çıkmaz". Bu vaat, bugün var olan anahtarları arayarak test edilemez.

    Bu yüzden metadata'ya SAHTE alanlar enjekte edip projeksiyonu çalıştırıyoruz.
    Bir alt modelin `extra="ignore"` ayarı `extra="allow"` yapılırsa ya da
    `ModelInfoResponse` metadata'yı olduğu gibi yansıtmaya başlarsa burası
    kırmızıya döner.
    """
    probe_metadata = copy.deepcopy(bundle.metadata)

    # Beyaz listede OLMAYAN alanlar, üç ayrı iç içe seviyeye serpiştiriliyor.
    probe_metadata["_internal_note"] = "SIZINTI-KOK"
    probe_metadata["feature_influence"]["_debug_dump"] = "SIZINTI-INFLUENCE"
    probe_metadata["feature_influence"]["features"]["age"]["_raw_tree_paths"] = "SIZINTI-ENTRY"
    probe_metadata["feature_influence"]["summary"]["_internal_rank"] = "SIZINTI-SUMMARY"
    probe_metadata["fairness"]["_reviewer_private_note"] = "SIZINTI-FAIRNESS"
    probe_metadata["fairness"]["protected_attributes_used_as_features"][0]["_ticket"] = (
        "SIZINTI-ATTRIBUTE"
    )
    probe_metadata["dataset"]["source_file"] = "C:/sunucu/gizli/yol/insurance_claims.csv"
    probe_metadata["metrics"]["_internal_holdout_auc"] = "SIZINTI-METRIC"

    probe = ModelBundle(bundle.pipeline, probe_metadata)
    rendered = json.dumps(
        model_info(probe).model_dump(by_alias=True), ensure_ascii=False, default=str
    )

    assert "SIZINTI" not in rendered, "beyaz liste iç içe bir anahtarı sızdırdı"
    assert "gizli" not in rendered, "sunucu tarafı dosya yolu sızdı"
    assert "source_file" not in rendered

    # Enjeksiyon gerçekten yanıtın DOKUNDUĞU yerlere yapılmış olmalı — aksi
    # hâlde test hiçbir şey kanıtlamadan yeşil kalırdı.
    assert "insured_sex" in rendered
    assert "_raw_tree_paths" in json.dumps(probe_metadata["feature_influence"]["features"]["age"])


def test_model_info_projection_covers_training_ranges_and_defaults(
    bundle: ModelBundle,
) -> None:
    """`training_ranges` ve `defaults` dallarına da canary enjekte edilir.

    Codex C-6: bir üstteki test dört metadata dalını kapsıyordu ama bu ikisini
    atlıyordu. `TrainingRangeInfo` / `DefaultInfo` bugün `extra="ignore"` — yani
    kod güvenli. Ancak kapsanmayan bir dal, "beyaz liste her yerde çalışıyor"
    iddiasını test edilmemiş bırakır: biri `extra="allow"`a dönerse ve metadata
    ileride oraya hassas bir alan koyarsa testler yeşilken sızıntı olur.
    """
    probe_metadata = copy.deepcopy(bundle.metadata)

    a_range = next(iter(probe_metadata["training_ranges"]))
    a_default = next(iter(probe_metadata["defaults"]))
    probe_metadata["training_ranges"][a_range]["_raw_column_sample"] = "SIZINTI-RANGE"
    probe_metadata["defaults"][a_default]["_source_row_id"] = "SIZINTI-DEFAULT"

    probe = ModelBundle(bundle.pipeline, probe_metadata)
    rendered = json.dumps(
        model_info(probe).model_dump(by_alias=True), ensure_ascii=False, default=str
    )

    assert "SIZINTI" not in rendered
    # Dallar yanıtta gerçekten var — enjeksiyon boşluğa yapılmadı.
    assert a_range in json.loads(rendered)["training_ranges"]
    assert a_default in json.loads(rendered)["defaults"]


# --------------------------------------------------------------------------- #
# 20) Codex ikinci görüşünün açtığı fail-fast boşlukları
# --------------------------------------------------------------------------- #


def test_body_limit_applies_to_endpoints_that_never_read_the_body(
    client: TestClient,
) -> None:
    """Gövdesini okumayan endpoint'lerde de gövde sınırı uygulanmalı (Codex C-1).

    BULUNAN AÇIK: sınırın akış bazlı kolu, uygulamanın `receive()` çağırmasına
    bağlıydı. `/health` ve `/model-info` istek gövdesini hiç okumaz, dolayısıyla
    `Content-Length` göndermeyen (chunked) bir istemci bu endpoint'lere sınırsız
    gövde akıtabiliyordu. Ölçüldü: 160 KB chunked gövde ile `GET /health` -> 200.

    `Content-Length` VARSA ucuz yol zaten yakalıyordu; açık yalnızca chunked
    yolundaydı — yani tam olarak başlığı hiç göndermeyen istemcide.
    """

    def chunks() -> Any:
        for _ in range(20):
            yield b"x" * 8192  # toplam 160 KB, Content-Length YOK

    for path in ("/health", "/model-info"):
        response = client.request("GET", path, content=chunks())
        assert response.status_code == 413, (
            f"{path} 160 KB chunked gövdeyi kabul etti ({response.status_code}) — "
            "gövde sınırı bu endpoint'te uygulanmıyor"
        )

    # Normal istekler etkilenmemeli: gövdesiz GET hâlâ çalışıyor.
    assert client.get("/health").status_code == 200
    assert client.get("/model-info").status_code == 200


def test_display_names_out_of_order_fails_fast(bundle: ModelBundle) -> None:
    """`transformed_display_names` PERMÜTE edilirse uygulama açılmamalı (Codex C-3).

    BULUNAN AÇIK: `_verify_feature_alignment` yalnızca uzunluğa ve `cat__` /
    `remainder__` önekine bakıyordu; ikisi de adların SIRASI hakkında hiçbir şey
    söylemez. Aynı adları farklı sırada içeren bir metadata ile servis sorunsuz
    açılıyor, skorlar doğru kalıyor, ama `/predict` her SHAP katkısını YANLIŞ
    feature'a atfediyordu.

    `_verify_shap_additivity` bunu yakalayamaz — toplam değişmediği için eşitlik
    korunur. Testler yakalıyordu, açılış yakalamıyordu; yani bozuk bir artefaktla
    production'a çıkmak mümkündü. Bu, demonun tam da satmaya çalıştığı şeyin
    (açıklanabilirlik) sessizce yalan söylemesi olurdu.
    """
    broken = copy.deepcopy(bundle.metadata)
    names = broken["feature_list"]["transformed_display_names"]
    names[0], names[1] = names[1], names[0]  # sadece SIRA bozuluyor

    probe = ModelBundle(bundle.pipeline, broken)
    with pytest.raises(ArtifactError, match="kolon sırasıyla eşleşmiyor"):
        probe._verify_feature_alignment()

    # Sağlamlık: bozulmamış metadata aynı kontrolden geçmeli.
    ModelBundle(bundle.pipeline, copy.deepcopy(bundle.metadata))._verify_feature_alignment()


def test_missing_positive_class_fails_fast_instead_of_guessing(
    bundle: ModelBundle,
) -> None:
    """`classes_` içinde pozitif etiket yoksa tahmin edilmemeli (Codex C-2).

    BULUNAN AÇIK: eski kod etiketi bulamayınca "son sınıf"a düşüyordu. Bu
    varsayım yanlış olduğunda hem `predict_proba` sütunu hem SHAP ekseni AYNI
    ANDA kayar — dolayısıyla `_verify_shap_additivity` de aynı yanlış ekseni
    kullanır ve eşitliği sağlar. Yani hata kendi doğrulamasını atlatıp "ters
    işaretli ama tutarlı" bir servis açardı.
    """

    class _WrongLabels:
        classes_ = np.array([0, 2])

    class _NoLabels:
        pass

    with pytest.raises(ArtifactError, match="pozitif sınıf"):
        _positive_class_index(_WrongLabels())

    with pytest.raises(ArtifactError, match="classes_"):
        _positive_class_index(_NoLabels())

    # Gerçek artefakt etkilenmemeli: pozitif sınıf 1, indeksi de doğru bulunuyor.
    assert _positive_class_index(bundle.pipeline.named_steps["model"]) == 1
