"""Faz 2 — guardrail (OOD tespiti) testleri.

TEST FELSEFESİ
--------------
Guardrail'in tehlikesi, "çalışıyor" görünürken hiçbir şey yapmamasıdır: her
istekte boş liste döndüren bir fonksiyon da mutlu mesut yeşil kalır. Bu yüzden
buradaki testler iki yönü birden bağlar — uyarı ÇIKMASI gereken yerde çıkıyor
mu, ve çıkmaMASI gereken yerde susuyor mu.

Ayrıca guardrail'in eşikleri KODA GÖMÜLÜ OLMAMALI. Gömülü olsaydı model
yeniden eğitildiğinde uyarılar sessizce yalan söylemeye başlardı; bunu bir
metadata mutasyonuyla test ediyoruz.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.guardrails import Guardrail
from app.model import ModelBundle
from app.schemas import PREDICT_REQUEST_EXAMPLE, PredictRequest


@pytest.fixture(scope="session")
def guardrail(bundle: ModelBundle) -> Guardrail:
    """Uygulamanın gerçekten kullandığı guardrail nesnesi."""
    return bundle.guardrail


# --------------------------------------------------------------------------- #
# 1) Temel davranış
# --------------------------------------------------------------------------- #


def test_values_inside_the_training_range_produce_no_warning(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Eğitim aralığının İÇİNDEKİ hiçbir değer uyarı üretmemeli.

    Her sayısal alan için aralığın ORTA noktası deneniyor: guardrail'in
    "her şeye uyarı basan" bir hâle düşmediğinin kanıtı.
    """
    ranges = metadata["training_ranges"]
    payload = {
        name: (spec["min"] + spec["max"]) / 2
        for name, spec in ranges.items()
        if spec["type"] == "numeric"
    }
    assert len(payload) == 18, "sayısal alan sayısı değişmiş"
    assert guardrail.check(payload) == []


def test_every_numeric_field_can_be_flagged(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """18 sayısal alanın HER BİRİ için OOD tetiklenebiliyor.

    Tek bir alanın kontrolü unutulsaydı ya da yanlış anahtarla okunsaydı
    (ör. `capital_gains` vs `capital-gains`) burası yakalar.
    """
    ranges = metadata["training_ranges"]
    numeric = {n: s for n, s in ranges.items() if s["type"] == "numeric"}

    for name, spec in numeric.items():
        above = guardrail.check({name: spec["max"] + 1})
        below = guardrail.check({name: spec["min"] - 1})
        assert above == [name], f"{name}: üst sınırın üstü işaretlenmedi ({above})"
        assert below == [name], f"{name}: alt sınırın altı işaretlenmedi ({below})"


def test_range_boundaries_are_inclusive(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Eğitimde GÖRÜLEN uç değerler uyarı üretmemeli.

    `min` ve `max` eğitim verisinde fiilen gözlemlenmiş değerlerdir; onları
    "dağılım dışı" saymak, modelin gördüğü veriyi görmedi diye işaretlemek olur.
    """
    ranges = metadata["training_ranges"]
    for name, spec in ranges.items():
        if spec["type"] != "numeric":
            continue
        assert guardrail.check({name: spec["min"]}) == [], f"{name}: min uyarı verdi"
        assert guardrail.check({name: spec["max"]}) == [], f"{name}: max uyarı verdi"


def test_constant_field_flags_anything_but_the_single_observed_value(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """`incident_year` eğitimde SABİT (min=max=2015) — 2015 dışı her şey OOD.

    Kenar durum: min ile max eşit olduğunda aralık tek bir noktaya iner.
    Bu veri setinin gerçeği (tüm olaylar 2015'te) ve guardrail'in en dar
    aralıkta da doğru çalıştığının kanıtı.
    """
    spec = metadata["training_ranges"]["incident_year"]
    assert spec["min"] == spec["max"] == 2015

    assert guardrail.check({"incident_year": 2015}) == []
    assert guardrail.check({"incident_year": 2014}) == ["incident_year"]
    assert guardrail.check({"incident_year": 2016}) == ["incident_year"]


# --------------------------------------------------------------------------- #
# 2) "Gönderilmedi" ile "aralık dışı" farkı
# --------------------------------------------------------------------------- #


def test_missing_fields_are_never_flagged(guardrail: Guardrail) -> None:
    """Gönderilmeyen alan uyarı üretmez — varsayılanla dolar.

    Bu ayrım olmasaydı boş bir istek (`{}`) 34 alanın hepsi için uyarı basardı
    ve alan tamamen değersizleşirdi. `metadata.defaults` değerleri tanım gereği
    eğitim aralığının içindedir; kullanıcının dokunmadığı bir alanı sorunlu
    göstermek yanlış olur.
    """
    assert guardrail.check({}) == []


def test_explicit_null_is_treated_as_missing_not_as_zero(guardrail: Guardrail) -> None:
    """Açıkça `null` gönderilen alan da "gönderilmedi" sayılır.

    `None`'ı sayısal karşılaştırmaya soksaydık TypeError alırdık; sessizce 0
    kabul etseydik `capital-loss` gibi negatif aralıklı alanlarda YANLIŞ uyarı
    üretirdik. `prepare_row()` de aynı ayrımı yapıyor (model.py K3).
    """
    assert guardrail.check({"age": None, "witnesses": None}) == []


def test_defaults_never_trigger_a_warning(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Varsayılan değerlerin hepsi eğitim aralığının içinde.

    Bu bir tutarlılık kontrolü: `defaults` medyan/moddan üretiliyor, dolayısıyla
    aralık içinde olmak ZORUNDA. Değilse guardrail ile defaults birbiriyle
    çelişiyor demektir ve boş bir istek bile uyarı üretirdi.
    """
    defaults = {name: spec["value"] for name, spec in metadata["defaults"].items()}
    assert guardrail.check(defaults) == []


# --------------------------------------------------------------------------- #
# 3) Kapsam sınırı: kategorik alanlar (Codex C-4)
# --------------------------------------------------------------------------- #


def test_categorical_fields_are_out_of_scope(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Kategorik alanlar guardrail'in kapsamında DEĞİL — ve bu bilinçli.

    `schemas.py`'daki `Literal` bilinmeyen kategoriyi 422 ile keser, yani
    kategorik bir OOD değeri buraya HİÇ ULAŞAMAZ. Kontrol yazsaydık ölü kod
    olurdu. Test bu sınırı açıkça kayda geçiriyor: kapsam daralırsa/genişlerse
    burası konuşur.
    """
    categorical = [
        name for name, spec in metadata["training_ranges"].items()
        if spec["type"] == "categorical"
    ]
    assert len(categorical) == 16

    for name in categorical:
        assert name not in guardrail.numeric_bounds
        # Kategorik alana çöp değer verilse bile guardrail susar.
        assert guardrail.check({name: "EGITIMDE-OLMAYAN-DEGER"}) == []


def test_unknown_category_is_rejected_by_the_api_not_by_the_guardrail(
    client: TestClient,
) -> None:
    """Kategorik OOD'nin gerçek davranışı: uyarı değil 422.

    Faz 4'teki UI banner'ı "kategorik OOD uyarısı gelir" varsayımıyla yazılırsa
    sessizce yanlış olur. Bu test o varsayımı baştan kırar.
    """
    response = client.post("/predict", json={"policy_state": "TX"})
    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["loc"] == ["body", "policy_state"]


# --------------------------------------------------------------------------- #
# 4) Sıralama: modelin gerçekten baktığı alanlar önce
# --------------------------------------------------------------------------- #


def test_influential_fields_are_listed_before_dead_ones(
    guardrail: Guardrail, metadata: dict[str, Any]
) -> None:
    """Etkili alanlar listenin başında, ölü alanlar sonunda.

    `umbrella_limit` (split_count=0) için aralık dışı bir değer skoru GERÇEKTEN
    etkilemez; `age` (canlı) için etkiler. İkisini karışık sırada listelemek
    okuyucuyu ikisinin aynı ağırlıkta olduğuna inandırırdı.
    """
    influence = metadata["feature_influence"]["features"]
    order = metadata["feature_list"]["pipeline_input_order"]

    # TEST SEÇİMİ ÖNEMLİ: ölü alanlar girdi sırasında canlı alanlardan ÖNCE
    # gelmeli, yoksa test sıralama olmadan da geçer (totolojik olur — ilk hâli
    # tam olarak buna düşmüştü ve mutasyon testinde yakalandı).
    #
    #   policy_deductable  sıra  4  ÖLÜ      witnesses   sıra 24  CANLI
    #   umbrella_limit     sıra  6  ÖLÜ      auto_year   sıra 31  CANLI
    #
    # Sıralama yapılmasaydı sonuç ["policy_deductable", "umbrella_limit",
    # "witnesses", "auto_year"] olurdu.
    for dead in ("policy_deductable", "umbrella_limit"):
        assert influence[dead]["has_influence"] is False
    for alive in ("witnesses", "auto_year"):
        assert influence[alive]["has_influence"] is True
    assert order.index("policy_deductable") < order.index("witnesses")
    assert order.index("umbrella_limit") < order.index("auto_year")

    warnings = guardrail.check(
        {
            "policy_deductable": 90_000,
            "umbrella_limit": 99_000_000,
            "witnesses": 9,
            "auto_year": 2050,
        }
    )
    assert warnings == ["witnesses", "auto_year", "policy_deductable", "umbrella_limit"], (
        f"canlı alanlar başa alınmadı: {warnings}"
    )


def test_ordering_is_deterministic(guardrail: Guardrail) -> None:
    """Aynı girdi -> aynı sıra. Sözlük ekleme sırasına bağlı kalmamalı."""
    first = guardrail.check({"age": 110, "witnesses": 9, "auto_year": 2050})
    second = guardrail.check({"auto_year": 2050, "witnesses": 9, "age": 110})
    assert first == second
    assert len(first) == 3


# --------------------------------------------------------------------------- #
# 5) Eşikler metadata'dan gelir, koda gömülü DEĞİL
# --------------------------------------------------------------------------- #


def test_thresholds_come_from_metadata_not_from_hardcoded_values(
    bundle: ModelBundle,
) -> None:
    """Metadata'daki aralık değişirse guardrail'in davranışı da değişmeli.

    Eşikler koda gömülü olsaydı model yeniden eğitildiğinde uyarılar sessizce
    yalan söylemeye başlardı — "bu değeri görmedim" derken aslında görmüş
    olurdu. Test, aralığı daraltıp daha önce temiz geçen bir değerin artık
    işaretlendiğini gösteriyor.
    """
    # Gerçek aralıkta 30 yaş tamamen normal (20-64).
    assert bundle.guardrail.check({"age": 30}) == []

    narrowed = copy.deepcopy(bundle.metadata)
    narrowed["training_ranges"]["age"]["min"] = 40
    narrowed["training_ranges"]["age"]["max"] = 50

    probe = Guardrail(narrowed)
    assert probe.check({"age": 30}) == ["age"]
    assert probe.check({"age": 45}) == []

    # Orijinal guardrail etkilenmemeli (paylaşılan durum sızıntısı yok).
    assert bundle.guardrail.check({"age": 30}) == []


# --------------------------------------------------------------------------- #
# 6) Sessiz geçiş tuzakları
# --------------------------------------------------------------------------- #


def test_nan_is_flagged_instead_of_silently_passing(guardrail: Guardrail) -> None:
    """NaN "aralık içinde" sayılıp sessizce geçmemeli.

    `nan < min` ve `nan > max` İKİSİ DE False döner. Özel kontrol olmasaydı NaN
    temiz bir istek gibi görünürdü — guardrail'in en sinsi sessiz hatası.
    """
    assert guardrail.check({"age": math.nan}) == ["age"]


def test_infinity_is_flagged(guardrail: Guardrail) -> None:
    """Sonsuz değer her üst sınırı aşar."""
    assert guardrail.check({"age": math.inf}) == ["age"]
    assert guardrail.check({"capital-loss": -math.inf}) == ["capital-loss"]


def test_booleans_are_not_treated_as_numbers(guardrail: Guardrail) -> None:
    """`True` sessizce 1 gibi karşılaştırılmamalı.

    Python'da `bool`, `int`in alt sınıfıdır: `isinstance(True, int)` doğrudur.
    Kontrol etmeseydik `witnesses: true` gönderen bir istemci "1 tanık" muamelesi
    görürdü. Bu katman karar vermez — Pydantic zaten 422 döndürür — ama guardrail
    doğrudan çağrıldığında da yanlış karşılaştırma yapmamalı.
    """
    assert guardrail.check({"witnesses": True}) == []
    assert guardrail.check({"age": False}) == []


def test_non_numeric_values_are_skipped_not_crashed(guardrail: Guardrail) -> None:
    """Sayısal alana string gelirse guardrail patlamamalı (Pydantic'in işi)."""
    assert guardrail.check({"age": "otuz"}) == []
    assert guardrail.check({"witnesses": [1, 2]}) == []


def test_unknown_keys_in_payload_are_ignored(guardrail: Guardrail) -> None:
    """Bilinmeyen anahtar guardrail'i düşürmemeli.

    API'de `extra="forbid"` bunu zaten kesiyor; guardrail HTTP'siz de
    çağrılabildiği için kendi başına dayanıklı olmalı.
    """
    assert guardrail.check({"boyle_bir_alan_yok": 999, "age": 110}) == ["age"]


# --------------------------------------------------------------------------- #
# 7) Uçtan uca: HTTP katmanıyla birlikte
# --------------------------------------------------------------------------- #


def test_warning_does_not_block_the_prediction(client: TestClient) -> None:
    """OOD uyarısı tahmini ENGELLEMEZ — skor yine döner.

    Guardrail'in tasarım kararı: reddetmek değil, işaretlemek. Sigorta eksperi
    için "reddedilen talep" ile "modelin emin olmadığı talep" aynı şey değil.
    """
    response = client.post("/predict", json={"witnesses": 9, "auto_year": 2050})
    assert response.status_code == 200
    body = response.json()

    assert set(body["out_of_distribution_warnings"]) == {"witnesses", "auto_year"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert len(body["shap_values"]) == 34


def test_training_extremes_pass_through_both_layers(
    client: TestClient, metadata: dict[str, Any]
) -> None:
    """Eğitimin uç değerleri hem Pydantic'ten hem guardrail'den temiz geçmeli.

    İki katmanın çakışmadığının kanıtı: Pydantic sınırı eğitim aralığının ÜST
    KÜMESİ olmak zorunda (schemas.py girişindeki 1. madde). Bir alanda Pydantic
    sınırı eğitim aralığının içine düşseydi, modelin gördüğü bir değeri 422 ile
    reddederdik.
    """
    ranges = metadata["training_ranges"]
    numeric = {n: s for n, s in ranges.items() if s["type"] == "numeric"}

    for bound in ("min", "max"):
        payload = {name: spec[bound] for name, spec in numeric.items()}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"{bound} uçları reddedildi: {response.text}"
        assert response.json()["out_of_distribution_warnings"] == []


def test_shap_values_are_still_correct_when_a_warning_fires(
    client: TestClient, bundle: ModelBundle
) -> None:
    """Uyarı üretmek tahmin yolunu bozmamalı.

    Guardrail `predict()` içine eklendi; SHAP toplanabilirliği hâlâ geçerli
    olmalı (yani guardrail hesap yoluna karışmıyor).
    """
    payload = {"witnesses": 9, "age": 110, "incident_severity": "Major Damage"}
    body = client.post("/predict", json=payload).json()

    assert body["out_of_distribution_warnings"]
    margin = sum(item["value"] for item in body["shap_values"])
    margin += body["shap_values"][0]["base_value"]
    reconstructed = 1.0 / (1.0 + math.exp(-margin))
    assert reconstructed == pytest.approx(body["fraud_probability"], abs=1e-9)

    # Doğrudan model katmanı da aynı uyarıları vermeli.
    direct = bundle.predict(PredictRequest(**payload).model_dump(by_alias=True))
    assert direct["out_of_distribution_warnings"] == body["out_of_distribution_warnings"]


def test_example_request_from_claude_md_is_clean(client: TestClient) -> None:
    """CLAUDE.md'deki örnek istek tamamen eğitim aralığının içinde."""
    body = client.post("/predict", json=PREDICT_REQUEST_EXAMPLE).json()
    assert body["out_of_distribution_warnings"] == []


# --------------------------------------------------------------------------- #
# 8) TEK YÖNLÜ KÖRLÜK — guardrail'in HTTP üzerinden gerçek kapsamı
# --------------------------------------------------------------------------- #


# Pydantic ALT sınırı eğitim minimumuna eşit olan alanlar: bu alanlarda
# "eğitim aralığının altında" yönünde OOD API üzerinden HİÇ tetiklenemez,
# çünkü daha küçük bir değer zaten 422 alır.
#
# Bu bir hata DEĞİL, fiziksel gerçeğin sonucu: tanık sayısı 0'ın altına, araç
# sayısı 1'in altına inemez. Eğitim seti bu alanların fiziksel tabanına zaten
# değiyor. Sınırları gevşetmek, fiziksel olarak imkânsız girdileri modele
# sokmak olurdu (bkz. `schemas.py` girişindeki "TEK YÖNLÜ KÖRLÜK" notu).
BLIND_BELOW = {
    "months_as_customer",
    "capital-gains",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "injury_claim",
    "property_claim",
}

# Aynı durumun üst sınırdaki hâli.
BLIND_ABOVE = {"capital-loss", "incident_hour_of_the_day"}


def _pydantic_bounds(column: str) -> tuple[float, float]:
    """Alanın YAYINLANAN JSON Schema sınırları (frontend'in gördüğü sözleşme)."""
    properties = PredictRequest.model_json_schema()["properties"]
    branches = [
        branch
        for branch in properties[column].get("anyOf", [properties[column]])
        if branch.get("type") != "null"
    ]
    assert len(branches) == 1, f"'{column}' için beklenmeyen şema dalı"
    return float(branches[0]["minimum"]), float(branches[0]["maximum"])


def test_one_way_blindness_matches_the_documented_list(
    metadata: dict[str, Any],
) -> None:
    """Guardrail'in hangi yönde tetiklenemediği ÖLÇÜLÜR, varsayılmaz.

    `schemas.py` bu listeyi yorumda beyan ediyor. Beyan ile gerçek ayrışırsa
    (ör. bir alanın Pydantic sınırı değişirse) burası konuşur — aksi hâlde
    Faz 4'te "bu alanda neden hiç uyarı gelmiyor?" sorusu çok daha pahalıya
    cevaplanırdı.
    """
    ranges = metadata["training_ranges"]
    measured_below: set[str] = set()
    measured_above: set[str] = set()

    for column, spec in ranges.items():
        if spec["type"] != "numeric":
            continue
        low, high = _pydantic_bounds(column)
        if low == float(spec["min"]):
            measured_below.add(column)
        if high == float(spec["max"]):
            measured_above.add(column)

    assert measured_below == BLIND_BELOW, (
        "alt yönde kör alanlar listesi değişmiş — schemas.py'daki not güncellenmeli"
    )
    assert measured_above == BLIND_ABOVE, (
        "üst yönde kör alanlar listesi değişmiş — schemas.py'daki not güncellenmeli"
    )


def test_blind_direction_returns_422_instead_of_a_warning(client: TestClient) -> None:
    """Kör yönde uyarı DEĞİL 422 gelir — davranışın uçtan uca kanıtı.

    Guardrail'in "sessizce çalışmıyor" olduğu izlenimini önler: o yönde uyarı
    üretilmemesinin sebebi guardrail'in çalışmaması değil, isteğin ona hiç
    ulaşmamasıdır.
    """
    # witnesses eğitimde 0-3; 0'ın altı fiziksel olarak imkânsız.
    assert client.post("/predict", json={"witnesses": -1}).status_code == 422
    # Ama ÜST yönde guardrail devrede.
    body = client.post("/predict", json={"witnesses": 9})
    assert body.status_code == 200
    assert body.json()["out_of_distribution_warnings"] == ["witnesses"]


def test_fields_blind_in_both_directions_can_never_warn(client: TestClient) -> None:
    """`incident_hour_of_the_day` iki yönde de kör — hiçbir zaman uyarı veremez.

    Günün saati tanımı gereği 0-23'tür ve eğitim seti bu aralığın tamamını
    kapsıyor. Yani bu alan için OOD kavramı anlamsızdır. Faz 4'te bu alanın
    yanına "uyarı gelebilir" göstergesi konulmamalı.
    """
    assert "incident_hour_of_the_day" in BLIND_BELOW & BLIND_ABOVE

    for hour in (0, 12, 23):
        body = client.post("/predict", json={"incident_hour_of_the_day": hour}).json()
        assert body["out_of_distribution_warnings"] == []

    for invalid in (-1, 24):
        assert client.post(
            "/predict", json={"incident_hour_of_the_day": invalid}
        ).status_code == 422
