"""Model artefaktının yüklenmesi, şema normalizasyonu, tahmin ve SHAP.

NE YAPAR, NEDEN BÖYLE YAPAR
---------------------------

1) ARTEFAKT YOLU SABİT, KULLANICI GİRDİSİNDEN GELMEZ
   `MODELS_DIR` yalnızca `__file__`'dan türetilir. Yolu ortam değişkeninden ya
   da istekten almak, path traversal / keyfi pickle yükleme demektir — ve
   pickle yüklemek kod çalıştırmakla eşdeğerdir. Tek bir sabit yol, bu sınıf
   saldırıyı tamamen kapatır.

2) TEK SEFER YÜKLENİR, FAIL-FAST
   `ModelBundle.load()` uygulama açılışında (FastAPI `lifespan`) bir kez
   çağrılır. Dosya yoksa ya da artefakt metadata ile tutarsızsa uygulama HİÇ
   açılmaz. "Ayakta ama her isteğe 500 dönen" bir servis, açılmayan bir
   servisten daha kötüdür: healthcheck yeşil görünür, sorun deploy'dan çok
   sonra fark edilir.

   `shap.TreeExplainer` de burada bir kez kurulur. Her istekte kurmak, ağaç
   yapısını her seferinde yeniden ayrıştırmak demektir (pahalı ve gereksiz).

3) PIPELINE'IN DIŞINDA KALAN 4 ADIM (preprocessing_contract)
   `pipeline.pkl` imputasyon ve encoding'i kapsar ama ham şemadan pipeline
   girdisine geçişi KAPSAMAZ. `metadata.preprocessing_contract` bu adımları
   makine-okunur olarak listeler. API katmanında durumları:

     drop_columns          -> TRIVIAL. Atılacak kolonlar (PII, ham tarihler,
                              _c39, auto_model) API şemasında zaten YOK.
                              İstemci gönderemez; `extra="forbid"` reddeder.
     derive_year_features  -> TRIVIAL. API ham tarih değil doğrudan
                              `incident_year` / `policy_bind_year` kabul eder,
                              yani türetilecek bir şey yok (bkz. K2).
     question_mark_to_nan  -> GERÇEKTEN UYGULANIR. `prepare_row()` içinde.
     order_columns         -> GERÇEKTEN UYGULANIR. `prepare_row()` içinde.

   Son iki adım atlanırsa hata SESSİZDİR: "?" string'i encoder tarafından
   bilinmeyen kategori sayılıp -1'e kodlanır (imputer hiç devreye girmez),
   yanlış kolon sırası ise tamamen yanlış feature'lara bakan bir tahmin üretir.
   İkisi de exception fırlatmaz — bu yüzden sözleşme kodda değil, artefaktta
   tutulur ve `_verify_contract()` ile açılışta doğrulanır.

4) ENCODING HÂLÂ YASAK
   Buradaki hiçbir şey kategorik değeri sayıya çevirmez. Yapılan iş yalnızca
   ŞEMA NORMALİZASYONUDUR (eksik alanı doldur, "?" -> NaN, kolonları sırala).
   Sayıya çevirme işi baştan sona `Pipeline`'ın içindedir.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import threading
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.pipeline import Pipeline

from app.guardrails import Guardrail

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Sabitler
# --------------------------------------------------------------------------- #

# backend/app/model.py -> backend/models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Kaynak veri setinde "bilinmiyor" bu string ile kodlanmış.
QUESTION_MARK = "?"

# Modelin pozitif sınıfı (fraud_reported == 1 -> "Y").
POSITIVE_CLASS_LABEL = 1

# --- Risk eşikleri (K6) ---------------------------------------------------- #
#
# DİKKAT — OLASILIKLAR KALİBRE DEĞİLDİR.
# Model `scale_pos_weight ~= 3.04` ile eğitildi: azınlık (fraud) sınıfı 3 kat
# ağırlıklandırıldı. Bu, PR-AUC'yi iyileştirir ama `predict_proba` çıktısını
# YUKARI ŞİŞİRİR. Yani 0.71, "bu talebin %71 ihtimalle dolandırıcılık olduğu"
# anlamına GELMEZ; yalnızca "diğer taleplere göre yüksek sırada" demektir.
#
# Dolayısıyla aşağıdaki eşikler istatistiksel bir olasılık kesimi değil, bir
# TRİYAJ politikasıdır. Kolayca değiştirilebilir olmaları için modül düzeyi
# sabit tutuldular; iş kuralı değişirse tek yerden ayarlanır (ve /model-info
# bu değerleri buradan okuyup yayınlar, frontend kendi kopyasını tutmaz).
RISK_THRESHOLD_MEDIUM = 0.35
RISK_THRESHOLD_HIGH = 0.65
RISK_RULE_DESCRIPTION = (
    f"low: p < {RISK_THRESHOLD_MEDIUM} | "
    f"medium: {RISK_THRESHOLD_MEDIUM} <= p < {RISK_THRESHOLD_HIGH} | "
    f"high: p >= {RISK_THRESHOLD_HIGH}"
)
CALIBRATION_WARNING = (
    "Olasılıklar kalibre DEĞİLDİR. Model dengesiz sınıf ağırlığıyla "
    "(scale_pos_weight ~= 3.04) eğitildi; bu, azınlık sınıfının olasılıklarını "
    "sistematik olarak yukarı şişirir. fraud_probability mutlak bir olasılık "
    "olarak değil, talepleri önceliklendirmek için bir SIRALAMA skoru olarak "
    "okunmalıdır. Mutlak olasılık gerekiyorsa modelin ayrıca kalibre edilmesi "
    "(ör. isotonic / Platt) şarttır."
)

# `preprocessing_contract.caller_must_apply_before_predict` içinde beklediğimiz
# adımlar. Artefakt yeniden eğitilip yeni bir adım eklenirse (ör. yeni bir
# türetilmiş feature), API bunu sessizce atlamak yerine açılışta patlar.
EXPECTED_CONTRACT_STEPS = {
    "drop_columns",
    "derive_year_features",
    "question_mark_to_nan",
    "order_columns",
}


class ArtifactError(RuntimeError):
    """Artefakt eksik ya da metadata ile tutarsız — uygulama açılmamalı."""


# --------------------------------------------------------------------------- #
# SHAP çıktı şekli normalizasyonu (K7)
# --------------------------------------------------------------------------- #


def _positive_class_index(model: Any) -> int:
    """Pozitif sınıfın (`1`) `classes_` içindeki indeksi.

    Sabit `1` yazmıyoruz: sınıf sırası estimator'ın gördüğü etiketlere bağlıdır.

    SESSİZ FALLBACK YOK (Codex C-2). Önceki hâli, etiket bulunamazsa "son sınıf"a
    düşüyordu. O varsayım ikili problemlerde çoğu zaman doğru ama BU kodda
    tehlikeliydi: yanlış indeks hem `predict_proba` sütununu hem SHAP eksenini
    aynı anda kaydırır, dolayısıyla `_verify_shap_additivity` de aynı yanlış
    ekseni kullanıp EŞİTLİĞİ SAĞLAR. Yani hata, kendi doğrulamasını atlatarak
    "ters işaretli ama tutarlı" bir servis açardı.

    Artefakt `fraud_reported` alanını 0/1'e map'lediği için `classes_` bu
    projede her zaman `[0, 1]`'dir; buraya düşmek artefaktın beklenmedik şekilde
    değiştiği anlamına gelir ve doğru davranış açılmamaktır.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ArtifactError(
            "Modelde `classes_` yok; pozitif sınıf indeksi belirlenemiyor. "
            "Artefakt bir sklearn sınıflandırıcısı olmayabilir."
        )
    classes = np.asarray(classes)
    matches = np.flatnonzero(classes == POSITIVE_CLASS_LABEL)
    if not matches.size:
        raise ArtifactError(
            f"Modelin sınıf etiketleri arasında pozitif sınıf "
            f"({POSITIVE_CLASS_LABEL}) yok: classes_={classes.tolist()}. "
            "Hangi sütunun 'fraud' olduğu tahmin edilemez; bir varsayımla devam "
            "etmek olasılığı ve SHAP işaretlerini sessizce ters çevirirdi."
        )
    return int(matches[0])


def _select_positive_class(values: Any, base_values: Any, positive_index: int) -> tuple[np.ndarray, float]:
    """SHAP çıktısından pozitif sınıfın katkılarını ŞEKLİ VARSAYMADAN seçer.

    `TreeExplainer` ikili sınıflandırmada sürüme/backend'e göre farklı şekiller
    döndürebilir; hepsini tek biçime indiriyoruz:

        list/tuple            -> sınıf başına dizi, [positive_index] seçilir
        ndarray (n, f, c)     -> son eksen sınıf, [..., positive_index]
        ndarray (n, f)        -> ikili modelde zaten pozitif sınıfın log-odds'u

    Bu kurulumda (shap 0.52 + lightgbm 4.6, objective='binary') ölçülen şekil
    (1, 34) ve skaler base_value'dur — ama yanlış eksene indekslemek sessizce
    işaretleri ters çeviren bir hata olurdu, o yüzden şekli kontrol ediyoruz.
    Seçimin DOĞRU olduğu ayrıca toplanabilirlik ile kanıtlanır
    (bkz. `_verify_shap_additivity`).
    """
    if isinstance(values, (list, tuple)):
        values = values[positive_index]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:  # (n_rows, n_features, n_classes)
        array = array[..., positive_index]
    if array.ndim == 2:  # (n_rows, n_features) -> tek satır istiyoruz
        array = array[0]
    if array.ndim != 1:
        raise ArtifactError(f"Beklenmeyen SHAP değer şekli: {np.asarray(values).shape}")

    if isinstance(base_values, (list, tuple)):
        base_values = base_values[positive_index]
    base = np.asarray(base_values, dtype=float)
    if base.ndim == 0:
        base_value = float(base)
    elif base.ndim == 1:
        # Tek satır tahmin ediyoruz: boyut 1 ise satır ekseni, değilse sınıf ekseni.
        base_value = float(base[0]) if base.size == 1 else float(base[positive_index])
    else:  # (n_rows, n_classes)
        base_value = float(base[0, positive_index])

    return array, base_value


# --------------------------------------------------------------------------- #
# Risk etiketi
# --------------------------------------------------------------------------- #


def classify_risk(probability: float) -> str:
    """Olasılığı `low` / `medium` / `high` triyaj etiketine çevirir."""
    if probability >= RISK_THRESHOLD_HIGH:
        return "high"
    if probability >= RISK_THRESHOLD_MEDIUM:
        return "medium"
    return "low"


# --------------------------------------------------------------------------- #
# ModelBundle
# --------------------------------------------------------------------------- #


class ModelBundle:
    """Yüklenmiş pipeline + metadata + SHAP explainer'ı bir arada tutar.

    Neden tek nesne: üçü birbirine bağlıdır (explainer pipeline'ın modeline,
    feature adları metadata'ya bakar). Ayrı global değişkenler tutmak, birinin
    güncellenip diğerinin unutulmasına açık bir tasarım olurdu.
    """

    def __init__(self, pipeline: Pipeline, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata

        self.input_order: list[str] = metadata["feature_list"]["pipeline_input_order"]
        self.defaults: dict[str, Any] = {
            name: spec["value"] for name, spec in metadata["defaults"].items()
        }
        self.display_names: list[str] = metadata["feature_list"]["transformed_display_names"]

        self._preprocessor = pipeline.named_steps["preprocessor"]
        self._model = pipeline.named_steps["model"]
        self._positive_index = _positive_class_index(self._model)

        # `question_mark_to_nan` kolonları KODA GÖMÜLMEZ, sözleşmeden okunur.
        # Model yeniden eğitilip bu liste değişirse API otomatik uyar.
        self.question_mark_columns: list[str] = list(
            _contract_step(metadata, "question_mark_to_nan")["columns"]
        )

        # Faz 2 — OOD kontrolü. Aralıklar guardrail'in içine gömülmez,
        # metadata'dan okunur: model yeniden eğitilirse eşikler kendiliğinden
        # güncellenir (bkz. `guardrails.py`).
        self.guardrail = Guardrail(metadata)

        self.explainer = shap.TreeExplainer(self._model)

        # shap'in LightGBM dalı her çağrıda `explainer.expected_value`'ı yeniden
        # ATAR. Değer hep aynı olsa da bu bir paylaşılan-durum mutasyonudur;
        # FastAPI senkron endpoint'leri thread havuzunda koştuğu için eş zamanlı
        # istekler aynı explainer'a girer. Tek satırlık bir açıklama saniyenin
        # binde biri sürüyor, kilidin maliyeti ihmal edilebilir — belirsiz bir
        # yarış durumuna tercih edilir.
        self._shap_lock = threading.Lock()

    # -- yükleme ------------------------------------------------------------ #

    @classmethod
    def load(cls) -> ModelBundle:
        """Artefaktları sabit yoldan yükler ve tutarlılığını doğrular."""
        missing = [p.name for p in (PIPELINE_PATH, METADATA_PATH) if not p.is_file()]
        if missing:
            raise ArtifactError(
                f"Model artefaktı bulunamadı: {missing} (aranan dizin: {MODELS_DIR}). "
                "Önce `python backend/train_pipeline.py` çalıştırın."
            )

        pipeline = joblib.load(PIPELINE_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        bundle = cls(pipeline, metadata)
        bundle._verify_contract()
        bundle._verify_feature_alignment()
        bundle._verify_transform_equivalence()
        bundle._verify_shap_additivity()
        bundle._warn_on_library_version_drift()
        return bundle

    def _verify_transform_equivalence(self) -> None:
        """`preprocessor.transform` + `model.predict_proba` == `pipeline.predict_proba`.

        `predict()` performans için pipeline'ı iki adıma ayırıyor (bkz.
        `_transform`). Bu kısayolun tam Pipeline çağrısıyla BİT BAZINDA aynı
        sonucu verdiğini açılışta kanıtlıyoruz. `pytest.approx` değil, tam
        eşitlik: en ufak sapma bile ayrıştırmanın yanlış olduğu anlamına gelir
        (ör. pipeline'a ileride üçüncü bir adım eklenirse bu kontrol patlar).
        """
        frame = self.prepare_row({})
        via_pipeline = self.pipeline.predict_proba(frame)
        via_steps = self._model.predict_proba(self._transform(frame))
        if not np.array_equal(via_pipeline, via_steps):
            raise ArtifactError(
                "preprocessor+model ayrıştırması pipeline.predict_proba ile aynı sonucu "
                f"vermiyor: {via_pipeline!r} != {via_steps!r}. Pipeline'da beklenmeyen "
                "bir adım olabilir; predict() kısayolu güvenli değil."
            )

    def _warn_on_library_version_drift(self) -> None:
        """Çalışma zamanı sürümleri eğitim sürümlerinden saptıysa UYARIR.

        NEDEN PATLATMIYORUZ: `pipeline.pkl` bir pickle'dır ve içinde sklearn /
        lightgbm / numpy sınıfları serileşmiştir. Farklı bir sürüm çoğu zaman
        sorunsuz açar, bazen sessizce farklı davranır, nadiren hiç açılmaz.
        Her sapmada açılışı reddetmek Faz 7'de (Docker imajı) gereksiz yere
        deploy'u kilitlerdi. Ama sessiz kalmak da olmaz: sessiz sapma tam olarak
        "model neden farklı skor veriyor?" sorusunun en pahalı hâlidir.

        Uyarı `/health` yanıtına EKLENMEZ — API sözleşmesi dar kalır, bu bir
        operatör sinyalidir, istemci verisi değil.
        """
        recorded: dict[str, str] = self.metadata.get("library_versions", {})
        runtime = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        }
        drift = {
            name: (recorded[name], runtime[name])
            for name in runtime
            if name in recorded and recorded[name] != runtime[name]
        }
        if drift:
            details = ", ".join(
                f"{name}: eğitim={trained} çalışma={current}"
                for name, (trained, current) in sorted(drift.items())
            )
            logger.warning(
                "Kütüphane sürüm sapması tespit edildi (%d paket). pipeline.pkl bu "
                "sürümlerle üretilmedi; tahminler sessizce farklılaşabilir. %s",
                len(drift),
                details,
            )

    def _verify_contract(self) -> None:
        """Sözleşmedeki adımların tam olarak beklediklerimiz olduğunu doğrular."""
        steps = {
            step["step"]
            for step in self.metadata["preprocessing_contract"]["caller_must_apply_before_predict"]
        }
        if steps != EXPECTED_CONTRACT_STEPS:
            raise ArtifactError(
                "preprocessing_contract adımları beklenenden farklı. "
                f"beklenen={sorted(EXPECTED_CONTRACT_STEPS)} bulunan={sorted(steps)}. "
                "API katmanı bilmediği bir hazırlık adımını sessizce atlayamaz."
            )

        # `order_columns` sözleşmesi ile gerçek girdi sırası aynı olmalı.
        ordered = _contract_step(self.metadata, "order_columns")["columns"]
        if list(ordered) != list(self.input_order):
            raise ArtifactError("order_columns sözleşmesi pipeline_input_order ile uyuşmuyor.")

        unknown_qm = set(self.question_mark_columns) - set(self.input_order)
        if unknown_qm:
            raise ArtifactError(f"question_mark_to_nan'da bilinmeyen kolon(lar): {sorted(unknown_qm)}")

    def _verify_feature_alignment(self) -> None:
        """defaults / transformed adlar / kolon sırası birbirini tutuyor mu?"""
        if set(self.defaults) != set(self.input_order):
            raise ArtifactError("metadata.defaults, pipeline_input_order ile aynı kolon kümesi değil.")

        transformed = list(self._preprocessor.get_feature_names_out())
        if transformed != list(self.metadata["feature_list"]["transformed_order"]):
            raise ArtifactError(
                "Pipeline'ın ürettiği transformed kolon sırası metadata ile uyuşmuyor."
            )
        if len(transformed) != len(self.display_names):
            raise ArtifactError(
                "transformed_display_names uzunluğu transformed_order ile eşleşmiyor: "
                f"{len(self.display_names)} != {len(transformed)}"
            )
        # SHAP adlarında cat__/remainder__ öneki SIZMAMALI (K7).
        leaked = [n for n in self.display_names if n.startswith(("cat__", "remainder__"))]
        if leaked:
            raise ArtifactError(f"transformed_display_names'te ham önek kalmış: {leaked}")

        # AD EŞLEMESİ POZİSYON BAZINDA DOĞRULANIR (Codex C-3).
        #
        # Yukarıdaki iki kontrol yalnızca UZUNLUĞA ve ÖNEKE bakıyordu; ikisi de
        # adların SIRASI hakkında hiçbir şey söylemez. Metadata'daki liste
        # permüte edilse (aynı adlar, farklı sıra) uygulama sorunsuz açılıyor,
        # skorlar doğru kalıyor ama `/predict` her SHAP katkısını YANLIŞ
        # feature'a atfediyordu — bu demonun tam da satmaya çalıştığı şeyin
        # sessizce yalan söylemesi demek.
        #
        # `_verify_shap_additivity` bunu yakalayamaz: toplam değişmediği için
        # eşitlik korunur. Testler yakalıyordu (bkz.
        # `test_shap_feature_names_follow_the_transformed_column_order`) ama
        # açılış yakalamıyordu; yani bozuk bir artefaktla production'a çıkmak
        # mümkündü. Doğrusu, adları metadata'ya güvenerek DEĞİL pipeline'ın
        # kendi çıktısından türetip karşılaştırmak.
        expected_display = [name.split("__", 1)[-1] for name in transformed]
        if self.display_names != expected_display:
            mismatches = [
                f"pozisyon {position}: metadata={actual!r} pipeline={expected!r}"
                for position, (actual, expected) in enumerate(
                    zip(self.display_names, expected_display, strict=True)
                )
                if actual != expected
            ]
            raise ArtifactError(
                "transformed_display_names, pipeline'ın kolon sırasıyla eşleşmiyor "
                f"({len(mismatches)} pozisyon). SHAP katkıları yanlış feature adlarıyla "
                f"yayınlanırdı. İlk farklar: {mismatches[:5]}"
            )

    def _verify_shap_additivity(self) -> None:
        """Pozitif sınıf seçiminin DOĞRU olduğunu matematiksel olarak kanıtlar.

        SHAP'in temel özdeşliği: sum(shap) + base_value = modelin ham skoru
        (log-odds). Yanlış sınıfı/ekseni seçseydik bu eşitlik bozulurdu. Bunu
        açılışta bir kez, varsayılan satır üzerinde kontrol ediyoruz — hatayı
        production'da 34 ters işaretli SHAP değeri olarak görmektense burada
        açılış hatası olarak görmek çok daha ucuz.
        """
        frame = self.prepare_row({})
        transformed = self._transform(frame)
        probability = float(self._model.predict_proba(transformed)[0, self._positive_index])
        values, base_value = self._shap_for_transformed(transformed)

        margin = float(values.sum() + base_value)
        # sigmoid(margin) LightGBM'in binary objective'inde predict_proba'ya eşit.
        reconstructed = 1.0 / (1.0 + math.exp(-margin))
        if abs(reconstructed - probability) > 1e-6:
            raise ArtifactError(
                "SHAP toplanabilirlik kontrolü başarısız: "
                f"sigmoid(sum(shap)+base)={reconstructed:.9f} != predict_proba={probability:.9f}. "
                "Muhtemelen yanlış sınıf ekseni seçildi."
            )

    # -- girdi hazırlığı ---------------------------------------------------- #

    def prepare_row(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Doğrulanmış istek gövdesini pipeline'ın beklediği tek satırlık
        DataFrame'e çevirir.

        `payload` anahtarları metadata kolon adlarıdır (Pydantic
        `model_dump(by_alias=True)` sayesinde `capital-gains` tireli hâliyle
        gelir).

        Uygulanan sözleşme adımları:
          * question_mark_to_nan  -> "?" gönderilen alan NaN yapılır
          * order_columns         -> kolonlar `pipeline_input_order` sırasında

        İKİ FARKLI "EKSİK" AYRIMI (K3 vs K4):
          alan hiç gönderilmedi / null  -> metadata.defaults değeri (medyan/mod)
          alan "?" gönderildi           -> NaN, pipeline'ın SimpleImputer'ı
        Sonuç bu modelde aynı yere çıkabilir (imputer da mod kullanıyor) ama
        anlamları farklıdır ve doldurma stratejisi değişirse davranış da
        doğru şekilde ayrışır.
        """
        record: dict[str, Any] = {}
        for column in self.input_order:
            value = payload.get(column)
            if value is None:
                # K3 — alan yok: eğitim setinden gelen medyan/mod.
                value = self.defaults[column]
            elif column in self.question_mark_columns and value == QUESTION_MARK:
                # K4 — alan "bilinmiyor": pipeline imputer'ına devret.
                value = np.nan
            record[column] = value

        # `columns=` ile inşa etmek `order_columns` adımını garanti eder.
        return pd.DataFrame([record], columns=self.input_order)

    # -- tahmin ------------------------------------------------------------- #

    def _shap_for_transformed(self, transformed: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Dönüştürülmüş satır için pozitif sınıfın SHAP katkıları + base value.

        Girdi olarak HAM satırı değil, ZATEN DÖNÜŞTÜRÜLMÜŞ matrisi alır: bu
        sayede modelin skorladığı matris ile explainer'ın açıkladığı matris
        aynı nesnedir — "aynı olmalı" bir yorum değil, kodun yapısı.
        """
        with self._shap_lock:
            # `explainer(X)` (Explanation API) tercih edildi: `shap_values()`
            # LightGBM ikili modellerde her çağrıda bir UserWarning basıyor —
            # her istekte log kirliliği. İkisi sayısal olarak birebir aynı.
            explanation = self.explainer(transformed)
            values, base_value = _select_positive_class(
                explanation.values, explanation.base_values, self._positive_index
            )
        return values, base_value

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Hazırlanmış satırı pipeline'ın KENDİ preprocessor adımıyla dönüştürür.

        ELLE ENCODING YASAĞI İHLAL EDİLMİYOR: burada çalışan şey pipeline'ın
        içindeki `ColumnTransformer`'ın ta kendisi (imputer + OrdinalEncoder).
        Kendi encoding sözlüğümüzü yazmıyoruz, sadece pipeline'ın iki adımını
        ayrı ayrı çağırıyoruz.

        NEDEN AYRI ÇAĞIRIYORUZ: `pipeline.predict_proba(frame)` preprocessor'ı
        bir kez, ardından SHAP için ikinci kez çalıştırmak gerekiyordu — istek
        başına aynı dönüşüm iki kez. Bir kez dönüştürüp aynı matrisi hem modele
        hem explainer'a vermek yalnızca daha hızlı değil, aynı zamanda
        "açıklanan şey ile skorlanan şey aynıdır" garantisini kod seviyesinde
        kesinleştirir.

        `_verify_transform_equivalence()` bu ayrıştırmanın
        `pipeline.predict_proba` ile BİT BAZINDA aynı sonucu verdiğini açılışta
        doğrular; yani kısayol sessizce sapamaz.
        """
        return self._preprocessor.transform(frame)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Tek bir talebi skorlar ve SHAP açıklamasıyla birlikte açıklar.

        Dönen sözlük doğrudan `PredictResponse` şemasına uyar; HTTP katmanı
        hiçbir hesaplama yapmaz.
        """
        frame = self.prepare_row(payload)
        transformed = self._transform(frame)

        probability = float(self._model.predict_proba(transformed)[0, self._positive_index])
        values, base_value = self._shap_for_transformed(transformed)

        if values.shape[0] != len(self.display_names):
            raise ArtifactError(
                f"SHAP {values.shape[0]} değer döndürdü, {len(self.display_names)} bekleniyordu."
            )

        # Tüm feature'lar döner (frontend kaçını göstereceğine kendi karar
        # verir), abs(value) azalan sırada — en etkili katkı ilk sırada.
        shap_values = [
            {"feature": name, "value": float(value), "base_value": base_value}
            for name, value in zip(self.display_names, values, strict=True)
        ]
        shap_values.sort(key=lambda item: abs(item["value"]), reverse=True)

        return {
            "fraud_probability": probability,
            "risk_level": classify_risk(probability),
            "shap_values": shap_values,
            # OOD kontrolü HAM payload üzerinde yapılır, `frame` üzerinde değil:
            # `prepare_row()` eksik alanları varsayılanla doldurduğu için
            # dönüştürülmüş satırda "kullanıcı bunu gönderdi mi?" bilgisi
            # kaybolur. Guardrail'in cevaplaması gereken soru tam olarak budur.
            "out_of_distribution_warnings": self.guardrail.check(payload),
        }


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #


def _contract_step(metadata: dict[str, Any], name: str) -> dict[str, Any]:
    """`preprocessing_contract` içinden adı verilen adımı getirir."""
    for step in metadata["preprocessing_contract"]["caller_must_apply_before_predict"]:
        if step["step"] == name:
            return step
    raise ArtifactError(f"preprocessing_contract'ta '{name}' adımı yok.")
