"""Faz 0 — Pipeline konsolidasyonu.

Bu script, notebook'lara dağılmış olan preprocessing (02_preprocessing_fe.ipynb)
ve model eğitimi (03_lightgbm.ipynb) adımlarını TEK bir sklearn `Pipeline`
içinde birleştirir ve iki artefakt üretir:

    backend/models/pipeline.pkl    -> preprocessor + model, tek dosya
    backend/models/metadata.json   -> defaults, training_ranges, metrics, versions

Neden tek Pipeline?
    Eski kurulumda `preprocessor.pkl` ve `best_model_lgb.pkl` ayrı dosyalardı ve
    servis katmanı ikisini elle sırayla çağırmak zorundaydı. Bu, "transform'u
    atlama" / "elle encoding sözlüğü yazma" hatalarına açık bir tasarım. Tek
    Pipeline ile *imputasyon + encoding* modelin ayrılmaz parçası olarak taşınır:
    çağıran taraf artık kendi encoding sözlüğünü yazamaz, yazmasına gerek de yok.

Pipeline'ın SINIRI (önemli — burayı yanlış okumak Faz 1'i bozar):
    Pipeline HER ŞEYİ kapsamaz. `pipeline.predict_proba(df)` çağrısındaki `df`,
    39 kolonluk HAM CSV satırı DEĞİL, `load_raw_frame()` temizliğinden geçmiş
    34 kolonluk çerçevedir (`metadata.feature_list.pipeline_input_order`).

    Pipeline'ın İÇİNDE (otomatik, çağıran hiçbir şey yapmaz):
      - kategorik NaN imputasyonu (`SimpleImputer(strategy='most_frequent')`)
      - kategorik -> sayısal encoding (`OrdinalEncoder`, bilinmeyen -> -1)
      - sayısal kolonlar passthrough (LightGBM NaN'ı native olarak taşır)

    Pipeline'ın DIŞINDA (`load_raw_frame()` içinde; API katmanı bunu KENDİSİ
    yapmak zorunda). Bu adımlar `build_preprocessing_contract()` tarafından
    makine-okunur biçimde `metadata.json -> preprocessing_contract` altına da
    yazılır; Faz 1 dokümana değil o alana bakmalı:
      1. `DROP_COLS` kolonlarını at (PII + ham tarihler + `_c39` + `auto_model`)
      2. `incident_date` -> `incident_year`, `policy_bind_date` ->
         `policy_bind_year` yıl türetmesi
      3. `QUESTION_MARK_COLS` içinde `"?"` -> `NaN` normalizasyonu.
         DİKKAT: Bunu API katmanı yapmazsa `"?"` string'i pipeline'a *geçerli
         bir kategori* gibi girer; encoder onu bilinmeyen sayıp -1'e kodlar ve
         imputer devreye GİRMEZ. Sessiz, sonucu bozan bir hata olur.
      4. Kolonları `pipeline_input_order` sırasına getir

    Yani Faz 1'deki `model.py` "pkl'i yükle, request'i doğrudan ver" diyemez;
    yukarıdaki 4 adımı uygulayan ince bir hazırlık katmanı yazmak zorundadır.
    Bu katman ENCODING YAPMAZ (o hâlâ yasak) — sadece şema normalizasyonu yapar.

Kritik: Pipeline HAM (temizlenmiş ama transform EDİLMEMİŞ) X_train üzerinde fit
edilir. Önceden transform edilmiş bir matris üzerinde fit etmek, preprocessor'ı
pipeline'ın dışında bırakır ve konsolidasyonun bütün amacını boşa çıkarır.

Kullanım:
    python backend/train_pipeline.py
    python backend/train_pipeline.py --data path/to/insurance_claims.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import shap
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# --------------------------------------------------------------------------- #
# Sabitler — notebook 02 & 03 ile birebir aynı tutuldu.
# --------------------------------------------------------------------------- #

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "insurance_claims.csv"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "models"

TARGET = "fraud_reported"
TARGET_MAP = {"Y": 1, "N": 0}

# PII + model için anlamsız / sızıntı riski taşıyan kolonlar.
# `policy_number`, `insured_zip`, `incident_location` doğrudan PII'dır ve
# BURADA drop edildikleri için defaults/metadata'ya da hiç girmezler.
# `incident_date` / `policy_bind_date` ham hâlleriyle drop edilir; bilgileri
# aşağıda `incident_year` / `policy_bind_year` olarak korunur.
# `_c39` Kaggle CSV'sindeki boş artık kolondur (bazı kopyalarda hiç yok).
# `auto_model` yüksek kardinaliteli ve `auto_make` ile büyük ölçüde eş anlamlı.
DROP_COLS = [
    "policy_number",
    "insured_zip",
    "incident_location",
    "incident_date",
    "policy_bind_date",
    "_c39",
    "auto_model",
]

# Ham tarih kolonlarından türetilen yıl feature'ları. Türetme Pipeline'ın
# DIŞINDA (load_raw_frame) yapılır; API katmanı da aynısını yapmak zorunda.
DATE_YEAR_FEATURES = {
    "incident_date": "incident_year",
    "policy_bind_date": "policy_bind_year",
}

# Bu kolonlarda eksik değer "?" string'i olarak kodlanmış.
QUESTION_MARK_COLS = ["property_damage", "police_report_available", "collision_type"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Optuna (notebook 03) tarafından bulunan en iyi parametreler. Faz 0 bir
# konsolidasyon fazıdır, modelleme fazı değil: HPO yeniden koşulmaz, bilinen
# en iyi parametrelerle deterministik refit yapılır.
LGBM_PARAMS = {
    "n_estimators": 173,
    "learning_rate": 0.010634,
    "num_leaves": 37,
    "min_child_samples": 93,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


# --------------------------------------------------------------------------- #
# Fairness / model card — korunan özellikler
# --------------------------------------------------------------------------- #
#
# Bu blok BİLİNÇLİ bir ürün kararının kaydıdır: model bu feature'larla eğitildi
# ve öyle kalıyor (yeniden eğitim yok). Ama sigorta/fintech bağlamında korunan
# nitelikle karar vermek düzenlemeye tabidir; artefaktın kendisi bunu sessizce
# taşımak yerine açıkça beyan eder.
#
# Serbest metin değil YAPILANDIRILMIŞ veri: Faz 5'teki model card sayfası bunu
# `GET /model-info` üzerinden okuyup tabloya basacak.

# Doğrudan korunan/hassas nitelikler (model feature'ı olarak KULLANILIYOR).
PROTECTED_ATTRIBUTES = [
    {
        "feature": "insured_sex",
        "basis": "sex",
        "severity": "high",
        "rationale": (
            "Cinsiyet, çoğu yargı bölgesinde doğrudan korunan niteliktir. Model "
            "feature'ı olarak kullanılıyor; dolandırıcılık skoru cinsiyete göre "
            "değişebilir."
        ),
    },
    {
        "feature": "age",
        "basis": "age",
        "severity": "high",
        "rationale": (
            "Yaş korunan niteliktir. Sigorta fiyatlamasında bazı yargı "
            "bölgelerinde aktüeryal gerekçeyle izinli olabilir, ancak "
            "dolandırıcılık *şüphesi* skorlamasında aynı muafiyet varsayılamaz."
        ),
    },
    {
        "feature": "insured_relationship",
        "basis": "marital_and_familial_status",
        "severity": "medium",
        "rationale": (
            "Değerler (husband/wife/unmarried/own-child/...) medeni ve ailevi "
            "durumu doğrudan açık eder; bu birçok yargı bölgesinde korunan "
            "niteliktir. Ayrıca cinsiyet için güçlü bir vekildir "
            "(husband/wife -> insured_sex)."
        ),
    },
    {
        "feature": "insured_education_level",
        "basis": "socioeconomic_proxy",
        "severity": "medium",
        "rationale": (
            "Klasik anlamda korunan sınıf değil, ancak sosyoekonomik köken ve "
            "dolaylı olarak ırk/ulusal köken için iyi belgelenmiş bir vekildir. "
            "Ayrımcılık denetiminde vekil değişken olarak ele alınmalıdır."
        ),
    },
]

# Doğrudan korunan nitelik değil ama denetimde vekil (proxy) riski taşıyanlar.
PROXY_RISK_ATTRIBUTES = [
    {
        "feature": "insured_occupation",
        "basis": "socioeconomic_proxy",
        "rationale": "Meslek; gelir, eğitim ve demografi için güçlü vekildir.",
    },
    {
        "feature": "insured_hobbies",
        "basis": "lifestyle_proxy",
        "rationale": (
            "SHAP'ta yüksek etkili çıkıyor; yaşam tarzı/kültürel köken vekili "
            "olabilir ve nedensel bir dolandırıcılık gerekçesi yoktur."
        ),
    },
    {
        "feature": "policy_state",
        "basis": "geographic_proxy",
        "rationale": "Coğrafi feature'lar redlining benzeri dolaylı ayrımcılık riski taşır.",
    },
    {
        "feature": "incident_state",
        "basis": "geographic_proxy",
        "rationale": "Coğrafi feature'lar redlining benzeri dolaylı ayrımcılık riski taşır.",
    },
    {
        "feature": "incident_city",
        "basis": "geographic_proxy",
        "rationale": "Coğrafi feature'lar redlining benzeri dolaylı ayrımcılık riski taşır.",
    },
]


def build_feature_influence(fitted_pipeline: Pipeline, display_names: list[str]) -> dict:
    """Her feature'ın eğitilmiş modeldeki ÖLÇÜLEN etkisi.

    NEDEN METADATA'DA, RUNTIME'DA DEĞİL:
    Split/gain sayıları eğitim anının olgusudur — eğitilmiş ağaçlar sabittir,
    bu sayılar her istekte yeniden hesaplanacak bir şey değil. Doğru evleri
    artefaktın yanı. Runtime'da hesaplamak hem israf hem de "hangi modelin
    sayısı?" belirsizliği yaratırdı.

    NEDEN ÖNEMLİ:
    Bir feature'ı modele VERMEK ile modelin onu KULLANMASI aynı şey değildir.
    LightGBM eğitim sırasında bir kolonda hiç bölünme yapmamışsa (`split_count`
    = 0) o kolon tahmini hiç etkilemez ve SHAP katkısı her zaman tam 0.0'dır.
    Bu ayrım hem ürün (kullanıcı formda bir alanı değiştirip sonucun
    kıpırdamadığını görüyor) hem de fairness beyanı açısından kritik.

    ANAHTARLAR API ALAN ADLARIDIR:
    `display_names` zaten `cat__`/`remainder__` öneki atılmış hâldir, yani
    `capital-gains` tireli hâliyle gelir — API şemasındaki alan adının ta
    kendisi. Frontend'in ayrıca bir eşleme tablosu tutması gerekmez.
    """
    booster = fitted_pipeline.named_steps["model"].booster_

    # Booster'ın kendi feature adları ile bizim temiz adlarımız POZİSYON
    # bazında eşleşmek zorunda; eşleşmezse bütün sayılar yanlış feature'a
    # yazılırdı ve bu sessiz bir hata olurdu.
    booster_names = [name.split("__", 1)[-1] for name in booster.feature_name()]
    if booster_names != list(display_names):
        raise ValueError(
            "Booster feature adları transformed_display_names ile eşleşmiyor; "
            "etki sayıları yanlış feature'lara yazılırdı."
        )

    split_counts = booster.feature_importance(importance_type="split")
    gains = booster.feature_importance(importance_type="gain")

    features: dict[str, dict] = {}
    for name, split_count, gain in zip(display_names, split_counts, gains):
        split_count = int(split_count)
        features[name] = {
            "split_count": split_count,
            "gain": float(gain),
            # Tanım: eğitilmiş ağaçlar bu kolonda en az bir kez bölündü mü?
            "has_influence": split_count > 0,
        }

    dead = sorted(name for name, info in features.items() if not info["has_influence"])

    return {
        "description": (
            "Eğitilmiş LightGBM booster'ından ölçülen feature etkisi. "
            "split_count = ağaçlarda bu kolon üzerinde yapılan bölünme sayısı, "
            "gain = bu bölünmelerin toplam kazancı, has_influence = split_count > 0."
        ),
        "measured_from": "lightgbm booster_.feature_importance(importance_type='split'|'gain')",
        "interpretation_note": (
            "has_influence=false olan bir feature bu EĞİTİLMİŞ modelde hiçbir "
            "tahmini etkilemez; SHAP katkısı her zaman tam 0.0'dır. Bu, "
            "feature'ın modele verilmediği anlamına GELMEZ — verildi, ağaçlar "
            "kullanmadı. Model yeniden eğitilirse (farklı veri, farklı seed, "
            "farklı hiperparametre) bu liste değişebilir."
        ),
        "summary": {
            "n_features": len(features),
            "n_with_influence": len(features) - len(dead),
            "n_without_influence": len(dead),
            "features_without_influence": dead,
        },
        "features": features,
    }


def build_fairness_section(
    cat_cols: list[str], num_cols: list[str], feature_influence: dict
) -> dict:
    """Model card'ın makine-okunur fairness bölümü.

    Feature adları eğitimde gerçekten kullanılan kolon listesine karşı
    doğrulanır: metadata'nın modelde olmayan bir feature'ı "korunan nitelik"
    diye ilan etmesi (ya da tersi) sessizce eskimesin.

    BEYAN İLE ÖLÇÜM AYRIŞTIRILDI:
    Eskiden bu bölümde yalnızca `used_as_model_feature: true` vardı. Teknik
    olarak doğru ama okuyucuyu yanlış yönlendiriyordu: model card "cinsiyet
    skoru etkileyebilir" derken, ölçüm "eğitilmiş ağaçlar bu kolonda hiç
    bölünme yapmamış" diyordu. Artık iki alan yan yana duruyor:

        used_as_model_feature -> feature modele VERİLDİ (beyan)
        has_influence         -> eğitilmiş ağaçlar onu KULLANDI (ölçüm)

    Sayılar `feature_influence`'tan gelir, yani booster'dan hesaplanır; elle
    yazılmış sabit değildir ve model değişince kendiliğinden güncellenir.
    """
    model_features = set(cat_cols) | set(num_cols)

    for entry in (*PROTECTED_ATTRIBUTES, *PROXY_RISK_ATTRIBUTES):
        if entry["feature"] not in model_features:
            raise ValueError(
                f"Fairness bildiriminde '{entry['feature']}' var ama model "
                "feature'ları arasında yok. Liste eğitimle senkron değil."
            )

    measured = feature_influence["features"]

    def with_measurement(entry: dict) -> dict:
        influence = measured[entry["feature"]]
        return {
            **entry,
            # BEYAN: feature modele girdi olarak verildi.
            "used_as_model_feature": True,
            # ÖLÇÜM: eğitilmiş modelde gerçekten kullanıldı mı?
            "split_count": influence["split_count"],
            "has_influence": influence["has_influence"],
        }

    return {
        "status": "declared_not_audited",
        "field_semantics": (
            "used_as_model_feature = feature modele VERİLDİ. "
            "has_influence = eğitilmiş ağaçlar bu kolonda gerçekten bölünme yaptı. "
            "İKİSİ AYNI ŞEY DEĞİLDİR: split_count=0 olan bir feature bu eğitilmiş "
            "modelde hiçbir tahmini etkilemez ve SHAP katkısı her zaman tam 0.0'dır. "
            "Bu ölçüm bir FAIRNESS DENETİMİ YERİNE GEÇMEZ: (1) model yeniden "
            "eğitilirse sonuç değişebilir, (2) vekil feature'lar (meslek, hobi, "
            "coğrafya) aynı sinyali dolaylı olarak geri taşıyabilir, (3) etkisi "
            "sıfır olmayan nitelikler için hiçbir grup bazlı metrik hesaplanmadı."
        ),
        "protected_attributes_used_as_features": [
            with_measurement(entry) for entry in PROTECTED_ATTRIBUTES
        ],
        "proxy_risk_attributes": [with_measurement(entry) for entry in PROXY_RISK_ATTRIBUTES],
        "audit_performed": False,
        "audit_metrics_computed": [],
        "intended_use": "demo_and_portfolio_only",
        "production_requirements": [
            (
                "Korunan nitelik başına grup bazlı fairness denetimi koş (ör. demografik "
                "parite farkı, eşitlenmiş fırsat / TPR-FPR farkı, kalibrasyon farkı)."
            ),
            (
                "Korunan niteliği düşürmenin ayrımcılığı gerçekten azaltıp azaltmadığını "
                "ölç: vekil feature'lar (meslek, hobi, coğrafya) sinyali geri taşıyabilir."
            ),
            (
                "İnsan gözden geçirmesi zorunlu: model çıktısı bir talebi tek başına "
                "reddetmek için değil, önceliklendirme/triyaj için kullanılmalı."
            ),
            (
                "Yerel düzenlemeyle uyumu doğrula (ör. AB AI Act yüksek riskli sistem "
                "yükümlülükleri, ABD eyalet sigorta ayrımcılık mevzuatı)."
            ),
        ],
        "notes": (
            "Model bu feature'larla eğitildi ve Faz 0'da bilinçli olarak öyle "
            "bırakıldı: amaç yayınlanan modeli SADIK biçimde yeniden üretmekti. "
            "Bu bölüm bir uyumluluk beyanı değil, bilinen riskin açık kaydıdır.\n\n"
            "ÖLÇÜM NOTU: bu eğitilmiş modelde bazı korunan/vekil niteliklerin "
            "split_count değeri 0'dır, yani ölçülen etkileri sıfırdır (her SHAP "
            "katkısı tam 0.0). Bu, modelin adil olduğu ya da ilgili nitelikle "
            "ayrımcılık yapmadığı ANLAMINA GELMEZ. Söylenebilecek tek şey, bu "
            "feature'ların bu spesifik eğitilmiş ağaç kümesindeki ölçülen "
            "etkisinin sıfır olduğudur. Etkisi sıfır olmayan nitelikler için "
            "hiçbir grup bazlı fairness metriği hesaplanmadı; audit_performed "
            "hâlâ false ve aşağıdaki production_requirements maddeleri aynen "
            "geçerlidir."
        ),
    }


def build_preprocessing_contract(pipeline_input_order: list[str]) -> dict:
    """Pipeline'ın DIŞINDA kalan hazırlığın makine-okunur sözleşmesi.

    Faz 1'de `model.py` bu adımları uygulamak zorunda; metadata'ya yazılması,
    "pipeline her şeyi hallediyor" yanılgısını artefakt seviyesinde kapatır.
    Kod sabitlerinden türetilir (elle yazılmaz) ki koddan asla ayrışmasın.
    """
    return {
        "summary": (
            "pipeline.pkl imputasyon ve encoding'i kapsar; ham CSV şemasından "
            "pipeline girdisine geçişi KAPSAMAZ. Çağıran (API katmanı) aşağıdaki "
            "adımları pipeline'a vermeden ÖNCE uygulamak zorundadır."
        ),
        "inside_pipeline": [
            "categorical_imputation: SimpleImputer(strategy='most_frequent')",
            "categorical_encoding: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
            "numeric_passthrough: imputasyon yok, LightGBM NaN'ı native taşır",
        ],
        "caller_must_apply_before_predict": [
            {
                "step": "drop_columns",
                "columns": DROP_COLS,
                "reason": "PII, ham tarihler, boş artık kolon ve yüksek kardinaliteli auto_model.",
            },
            {
                "step": "derive_year_features",
                "mapping": dict(DATE_YEAR_FEATURES),
                "reason": "Pipeline tarih parse etmez; yıl feature'ı dışarıda türetilir.",
            },
            {
                "step": "question_mark_to_nan",
                "columns": QUESTION_MARK_COLS,
                "reason": (
                    "Bu kolonlarda eksik değer '?' string'i olarak kodlanmış. "
                    "Dönüştürülmezse encoder '?' değerini bilinmeyen kategori sayıp "
                    "-1'e kodlar ve imputer devreye girmez -> sessiz hatalı tahmin."
                ),
            },
            {
                "step": "order_columns",
                "columns": pipeline_input_order,
                "reason": "Pipeline girdisi bu kolon adlarını ve sırasını bekler.",
            },
        ],
        "note_on_encoding_ban": (
            "Bu adımlar ŞEMA NORMALİZASYONUDUR, encoding değildir. Kategorik "
            "değerleri sayıya çevirmek hâlâ yalnızca pipeline'ın işidir."
        ),
    }


# --------------------------------------------------------------------------- #
# Veri hazırlığı
# --------------------------------------------------------------------------- #


def load_raw_frame(csv_path: Path) -> pd.DataFrame:
    """Ham CSV'yi okur ve notebook 02'deki temizliği uygular.

    DİKKAT: Notebook 02'de tasarlanan feature engineering (`vehicle_age`,
    `injury_ratio` vb.) burada YOKTUR — o kod bir markdown hücresinde kalmış,
    hiçbir zaman çalışmamıştı; yayınlanan model onlarsız eğitildi. Faz 0 o
    modeli SADIK biçimde yeniden üretir, yeni feature EKLEMEZ.

    Tek istisna `incident_year` / `policy_bind_year`: bunlar yeni bilgi değil,
    ham tarih kolonlarının modele verilebilir hâlidir ve orijinal eğitimde de
    vardı. (Not: `incident_year` bu veri setinde sabittir — tüm satırlar 2015 —
    yani modele bilgi katmaz; Faz 2 guardrail'i bunu hesaba katmalı.)

    Bu fonksiyonun tamamı Pipeline'ın DIŞINDADIR; API katmanı aynı adımları
    uygulamak zorunda (bkz. modül docstring'i + `build_preprocessing_contract`).
    """
    df = pd.read_csv(csv_path)

    # Tarih kolonlarından yıl türet, sonra ham tarihleri drop et.
    # Bunlar ZORUNLU: yoksa `incident_year`/`policy_bind_year` üretilemez ve
    # pipeline eksik kolonla çağrılır. Çıplak `KeyError` yerine hangi kolonun
    # eksik olduğunu söyleyen bir hata veriyoruz.
    missing_dates = [c for c in DATE_YEAR_FEATURES if c not in df.columns]
    if missing_dates:
        raise ValueError(
            f"CSV'de zorunlu tarih kolon(lar)ı yok: {missing_dates}. "
            f"Bunlardan {[DATE_YEAR_FEATURES[c] for c in missing_dates]} türetiliyor."
        )
    for date_col, year_col in DATE_YEAR_FEATURES.items():
        df[year_col] = pd.to_datetime(df[date_col]).dt.year

    present = [c for c in DROP_COLS if c in df.columns]
    absent = [c for c in DROP_COLS if c not in df.columns]
    if absent:
        print(f"[info] drop listesinde olup CSV'de bulunmayan kolonlar: {absent}")
    df = df.drop(columns=present)

    # "?" -> NaN. İmputasyon pipeline içinde (SimpleImputer) yapılır ki
    # train/serve arasında davranış farkı oluşmasın.
    # DROP_COLS ile aynı savunma: eksik kolonu doğrudan indekslemek `KeyError`
    # ile patlar; burada atlanır ve durum görünür kılınır. (Sessizce yutulmaz:
    # eksik kolon kategorik listesine hiç girmeyeceği için pipeline zaten
    # tutarlı kalır, ama operatör bunu bilmeli.)
    qm_present = [c for c in QUESTION_MARK_COLS if c in df.columns]
    qm_absent = [c for c in QUESTION_MARK_COLS if c not in df.columns]
    if qm_absent:
        print(f"[info] '?' normalizasyon listesinde olup CSV'de bulunmayan kolonlar: {qm_absent}")
    if qm_present:
        df[qm_present] = df[qm_present].replace("?", np.nan)

    if TARGET not in df.columns:
        raise ValueError(f"CSV'de hedef kolon '{TARGET}' yok.")
    df[TARGET] = df[TARGET].map(TARGET_MAP)
    if df[TARGET].isna().any():
        raise ValueError(f"{TARGET} kolonunda {TARGET_MAP} dışında değer var.")
    df[TARGET] = df[TARGET].astype("int64")

    return df


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Kategorik / sayısal kolonları ayırır.

    pandas 3'te metin kolonları `str` dtype'ına geçti; `include='object'` hâlâ
    geriye dönük uyumluluk için onları yakalıyor ama Pandas4Warning üretiyor.
    `include=['object', 'str']` aynı kolon kümesini uyarısız verir (doğrulandı).
    """
    feature_df = df.drop(columns=[TARGET])
    cat_cols = feature_df.select_dtypes(include=["object", "str"]).columns.tolist()
    num_cols = [c for c in feature_df.columns if c not in cat_cols]
    return cat_cols, num_cols


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def build_pipeline(cat_cols: list[str], scale_pos_weight: float) -> Pipeline:
    """preprocessor + model -> tek Pipeline.

    OrdinalEncoder'da `handle_unknown='use_encoded_value', unknown_value=-1`
    orijinalde YOKTU. Notebook bağlamında sorun değildi (encoder sadece eğitim
    verisini görüyordu) ama bir API için kabul edilemez: kullanıcı eğitimde
    görülmemiş bir kategori gönderdiğinde pipeline ValueError ile çöküyordu.
    Artık bilinmeyen kategori -1'e kodlanır; Faz 2'deki guardrail bu durumu
    `out_of_distribution_warnings` ile ayrıca kullanıcıya bildirecek.
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", cat_pipeline, cat_cols)],
        remainder="passthrough",
    )

    # ColumnTransformer'ın çıktısını DataFrame'e çeviriyoruz.
    #
    # Neden: varsayılan numpy çıktısında LightGBM'in sklearn sarmalayıcısı
    # kolonlara `Column_0..Column_33` gibi sahte isimler uydurup bunları
    # `feature_names_in_`e yazıyor. Sonra her `predict` çağrısında sklearn
    # "X does not have valid feature names, but LGBMClassifier was fitted with
    # feature names" uyarısı basıyor — FastAPI altında bu her istekte log
    # kirliliği demek. Ayrıca SHAP, feature adı olarak o anlamsız `Column_i`
    # etiketlerini görüyor.
    #
    # `set_output` ayarı estimator'ın üzerinde durur ve pickle ile birlikte
    # taşınır, yani serving tarafında global `sklearn.set_config` çağırmaya
    # gerek kalmaz. Tahminler üzerinde etkisi YOKTUR: numpy ve pandas
    # çıktısıyla eğitilen modellerin predict_proba sonuçları birebir aynı
    # (maksimum mutlak fark 0.0) — doğrulandı.
    preprocessor.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGBM_PARAMS)),
        ]
    )


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def _py(value):
    """numpy/pandas skalerlerini JSON-native tiplere çevirir."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    return value


def _require_finite(value, col: str, what: str) -> None:
    """metadata.json'a NaN/Infinity yazılmasını engeller.

    Neden `raise`, neden "sessizce 0 yaz" değil:
    `json.dumps` varsayılan olarak NaN'ı ÇIPLAK `NaN` token'ı olarak yazar. Bu
    geçerli JSON DEĞİLDİR: JavaScript'in `JSON.parse`'ı (yani frontend'in
    `/model-info` çağrısı) dosyayı tamamen reddeder. Python'ın `json.load`'u
    ise gevşek olduğu için hatayı sessizce yutar — yani sorun ancak frontend'de,
    en geç yerde patlar.

    Kategorik tarafta mod boşsa zaten `ValueError` fırlatılıyordu; sayısal
    tarafta hiçbir kontrol yoktu. Bu asimetriyi kapatıyoruz. Uydurma bir
    varsayılan yazmak yerine eğitimi durduruyoruz: tamamen boş bir sayısal
    kolon veri hatasıdır, artefaktla gizlenmemeli.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        raise ValueError(
            f"'{col}' kolonu için {what} sonlu bir sayı değil ({value!r}). "
            "Kolon büyük olasılıkla eğitim setinde tamamen boş. "
            "metadata.json geçerli JSON olmazdı; eğitim durduruldu."
        )


def build_defaults(X_train: pd.DataFrame, cat_cols: list[str]) -> dict:
    """API'de verilmeyen alanları doldurmak için kullanılacak varsayılanlar.

    Sayısal -> MEDYAN, kategorik -> MOD.
    Orijinal notebook her kolona `mode()` uyguluyordu; `total_claim_amount` gibi
    sürekli bir değişkende mod istatistiksel olarak anlamsızdır (çoğu zaman
    rastgele bir tekil gözlem). Medyan hem daha temsili hem aykırı değere
    dayanıklı.

    Değerler SADECE X_train'den hesaplanır — test verisi varsayılanlara sızmaz.

    Tam sayı kolonlarında medyan (n çift ise) .5 çıkabilir; değeri kolonun
    eğitimdeki dtype'ına yuvarlıyoruz ki JSON'dan geri yüklenen satır eğitim
    verisiyle birebir aynı tiplere sahip olsun ("tip kaybı olmamalı" gereği).
    """
    defaults: dict[str, dict] = {}

    for col in X_train.columns:
        series = X_train[col]
        dtype_name = str(series.dtype)

        if col in cat_cols:
            mode = series.mode(dropna=True)
            if mode.empty:
                raise ValueError(f"'{col}' için mod hesaplanamadı (tüm değerler NaN).")
            value = str(mode.iloc[0])
            strategy = "mode"
            dtype_name = "str"
        else:
            median = float(series.median())
            # Tamamen boş bir sayısal kolonda `median()` NaN döner ve bu
            # metadata.json'ı geçersiz JSON yapar (bkz. `_require_finite`).
            _require_finite(median, col, "medyan")
            if pd.api.types.is_integer_dtype(series):
                value = round(median)
            else:
                value = float(median)
            strategy = "median"

        defaults[col] = {"value": _py(value), "dtype": dtype_name, "strategy": strategy}

    # Güvenlik teyidi: PII kolonları drop edildi, defaults üzerinden sızamaz.
    leaked = [c for c in ("policy_number", "insured_zip", "incident_location") if c in defaults]
    if leaked:
        raise ValueError(f"PII alanı defaults'a sızmış: {leaked}")

    return defaults


def build_training_ranges(
    X_train: pd.DataFrame,
    fitted_pipeline: Pipeline,
    cat_cols: list[str],
    num_cols: list[str],
) -> dict:
    """Faz 2 guardrail'inin dayanacağı "eğitimde ne gördük" kaydı.

    Bu bilgi eğitim anında yakalanmazsa sonradan geri kazanılamaz (CSV
    kaybolabilir, split değişebilir), o yüzden metadata'ya yazılıyor.

    - Sayısal: X_train'deki min/max. Bunun dışındaki input ekstrapolasyondur.
    - Kategorik: fitted OrdinalEncoder'ın `categories_` listesi. Kasıtlı olarak
      ham X_train'deki unique'ler DEĞİL: encoder imputasyon SONRASI fit edilir,
      dolayısıyla `categories_` tam olarak "gerçek bir koda maplenen" değerler
      kümesidir. Bu listede olmayan her şey -1 (unknown) yolundan geçer, yani
      guardrail ile encoder birebir aynı gerçeği raporlar.
    """
    ranges: dict[str, dict] = {}

    for col in num_cols:
        series = X_train[col]
        col_min = _py(series.min())
        col_max = _py(series.max())
        # Boş sayısal kolonda min/max NaN olur -> geçersiz JSON (bkz.
        # `_require_finite`). Ayrıca NaN sınırlı bir aralık Faz 2 guardrail'ini
        # sessizce işlevsiz bırakırdı: `x < NaN` her zaman False, yani hiçbir
        # değer OOD sayılmazdı.
        _require_finite(col_min, col, "min")
        _require_finite(col_max, col, "max")
        ranges[col] = {
            "type": "numeric",
            "dtype": str(series.dtype),
            "min": col_min,
            "max": col_max,
        }

    encoder = fitted_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps[
        "encoder"
    ]
    encoder_cols = list(
        fitted_pipeline.named_steps["preprocessor"].transformers_[0][2]
    )
    if encoder_cols != cat_cols:
        raise ValueError("ColumnTransformer'daki kategorik kolon sırası beklenenden farklı.")

    for col, categories in zip(encoder_cols, encoder.categories_):
        ranges[col] = {
            "type": "categorical",
            "dtype": "str",
            "categories": sorted(str(c) for c in categories),
        }

    missing = set(X_train.columns) - set(ranges)
    if missing:
        raise ValueError(f"training_ranges'te eksik kolonlar: {sorted(missing)}")

    return ranges


def clean_transformed_names(preprocessor: ColumnTransformer) -> list[str]:
    """'cat__incident_severity' / 'remainder__witnesses' -> 'incident_severity'.

    SHAP çıktısının frontend'de okunabilir olması için gerekli.
    """
    return [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Faz 0 — pipeline konsolidasyonu")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    csv_path: Path = args.data.resolve()
    output_dir: Path = args.output_dir.resolve()
    if not csv_path.exists():
        print(f"[hata] veri dosyası bulunamadı: {csv_path}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Veri yükleniyor: {csv_path}")
    df = load_raw_frame(csv_path)
    cat_cols, num_cols = split_columns(df)
    print(f"      satır={len(df)}  kategorik={len(cat_cols)}  sayısal={len(num_cols)}")

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # Erken sınıf kontrolü. İki ayrı gizli patlama noktasını kapatır:
    #   - `train_test_split(stratify=y)`: tek sınıflı ya da bir sınıfı tek
    #     örnekli veride anlaşılması zor bir ValueError fırlatır.
    #   - `scale_pos_weight`: pozitif sınıf yoksa sıfıra bölme.
    # İkisi de ancak akışın ortasında patlar; burada ne olduğunu söyleyen tek
    # bir hatayla erken duruyoruz.
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            f"Hedef '{TARGET}' tek sınıflı: {class_counts.to_dict()}. "
            "İkili sınıflandırma için her iki sınıf da gerekli "
            f"(beklenen etiketler: {sorted(TARGET_MAP.values())})."
        )
    min_class_count = int(class_counts.min())
    # Stratified split'in her sınıftan hem train'e hem test'e örnek koyabilmesi
    # için sınıf başına en az 2 örnek şart.
    if min_class_count < 2:
        raise ValueError(
            f"Hedef '{TARGET}' sınıf dağılımı stratified split için yetersiz: "
            f"{class_counts.to_dict()}. Sınıf başına en az 2 örnek gerekir."
        )

    # Split, feature engineering'den ÖNCE ve pipeline fit'inden ÖNCE yapılır.
    # Imputer/encoder istatistikleri sadece X_train'den öğrenilir (leak yok).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    n_pos_train = int((y_train == 1).sum())
    if n_pos_train == 0:
        raise ValueError(
            "Eğitim setinde hiç pozitif örnek yok; scale_pos_weight hesaplanamaz. "
            f"test_size={TEST_SIZE} ile split çok küçük bir azınlık sınıfını "
            "tamamen test tarafına atmış olabilir."
        )
    scale_pos_weight = float((y_train == 0).sum() / n_pos_train)
    print(f"[2/6] Split: train={len(X_train)} test={len(X_test)}  scale_pos_weight={scale_pos_weight:.4f}")

    print("[3/6] Pipeline HAM X_train üzerinde fit ediliyor...")
    pipeline = build_pipeline(cat_cols, scale_pos_weight)
    pipeline.fit(X_train, y_train)

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    train_pr_auc = float(average_precision_score(y_train, train_proba))
    test_pr_auc = float(average_precision_score(y_test, test_proba))

    print(f"[4/6] Train PR-AUC: {train_pr_auc:.4f}")
    print(f"      Test  PR-AUC: {test_pr_auc:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, pipeline.predict(X_test), digits=3))

    print("[5/6] Metadata oluşturuluyor...")
    preprocessor = pipeline.named_steps["preprocessor"]
    pipeline_input_order = list(X_train.columns)
    transformed_display_names = clean_transformed_names(preprocessor)

    # Feature etkisi ölçümü fairness bölümünden ÖNCE hesaplanır: fairness
    # beyanı bu ölçüme dayanıyor (beyan edilen ile ölçülen ayrı alanlar).
    feature_influence = build_feature_influence(pipeline, transformed_display_names)
    n_dead = feature_influence["summary"]["n_without_influence"]
    print(
        f"      feature etkisi: {feature_influence['summary']['n_with_influence']} "
        f"etkili / {n_dead} etkisiz (split=0)"
    )
    if n_dead:
        print(f"      etkisiz: {', '.join(feature_influence['summary']['features_without_influence'])}")
    metadata = {
        "model_name": "insurance-fraud-detection",
        "model_version": "1.0.0",
        "algorithm": "LightGBM (LGBMClassifier)",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "source_file": csv_path.name,
            "n_rows": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        },
        "metrics": {
            "train_pr_auc": train_pr_auc,
            "test_pr_auc": test_pr_auc,
            "metric_name": "average_precision_score (PR-AUC)",
        },
        "feature_list": {
            # ADLANDIRMA UYARISI: bu liste eskiden `raw_input_order` deniyordu,
            # ama içeriği HAM CSV kolonları DEĞİL. Ham CSV 39 kolondur ve
            # `incident_date` / `policy_bind_date` / PII kolonlarını içerir;
            # buradaki 34 kolon ise `load_raw_frame()` temizliğinden SONRA
            # pipeline'ın gerçekten beklediği şemadır. Faz 1 bu listeye
            # güveneceği için ad ile içerik artık uyumlu.
            "pipeline_input_order": pipeline_input_order,
            "pipeline_input_order_description": (
                "pipeline.predict_proba()'ya verilecek DataFrame'in kolon adları ve "
                "sırası. Ham CSV şeması DEĞİLDİR: preprocessing_contract."
                "caller_must_apply_before_predict adımları uygulandıktan SONRAKİ "
                "şemadır."
            ),
            "categorical_features": cat_cols,
            "numeric_features": num_cols,
            # ColumnTransformer çıktısındaki kolon sırası (SHAP eşlemesi için).
            "transformed_order": list(preprocessor.get_feature_names_out()),
            "transformed_display_names": transformed_display_names,
            "dropped_columns": DROP_COLS,
            "derived_features": dict(DATE_YEAR_FEATURES),
            "target": TARGET,
        },
        "feature_influence": feature_influence,
        "preprocessing_contract": build_preprocessing_contract(pipeline_input_order),
        "fairness": build_fairness_section(cat_cols, num_cols, feature_influence),
        "defaults": build_defaults(X_train, cat_cols),
        "training_ranges": build_training_ranges(X_train, pipeline, cat_cols, num_cols),
        "model_params": {
            **{k: _py(v) for k, v in LGBM_PARAMS.items()},
            "scale_pos_weight": scale_pos_weight,
        },
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        },
    }

    print("[6/6] Artefaktlar yazılıyor...")
    pipeline_path = output_dir / "pipeline.pkl"
    metadata_path = output_dir / "metadata.json"
    joblib.dump(pipeline, pipeline_path)

    # `allow_nan=False`: son savunma hattı. Kolon bazlı kontroller
    # (`_require_finite`) hedefli ve okunabilir hata verir; bu ise metadata'nın
    # HERHANGİ bir yerinde NaN/Infinity kalırsa yazmayı tamamen engeller.
    # Varsayılan `allow_nan=True` bunları çıplak `NaN`/`Infinity` token'ı olarak
    # yazar — Python geri okuyabilir ama bu GEÇERLİ JSON DEĞİLDİR ve frontend'in
    # `JSON.parse`'ı dosyayı reddeder. Hata burada patlasın, tarayıcıda değil.
    serialized = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
    metadata_path.write_text(serialized, encoding="utf-8")

    print(f"      {pipeline_path}  ({pipeline_path.stat().st_size / 1024:.1f} KB)")
    print(f"      {metadata_path}  ({metadata_path.stat().st_size / 1024:.1f} KB)")
    print("\nTamam. Model artefaktı olarak pipeline.pkl tek başına yeterlidir; "
          "preprocessor.pkl / best_model_lgb.pkl / defaults.pkl artık gerekmiyor.")
    print("Ancak pipeline HAM CSV satırı kabul ETMEZ: çağıran taraf önce "
          "metadata.json -> preprocessing_contract adımlarını uygulamalı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
