"""API sözleşmesi — Pydantic v2 request/response modelleri.

NE YAPAR, NEDEN BÖYLE YAPAR
---------------------------
Bu dosya iki soruya cevap verir: "İstemci ne gönderebilir?" ve "Biz ne
döndürürüz?". Modelin nasıl çalıştığına dair hiçbir bilgi burada YOKTUR —
artefakt yükleme, imputasyon ve SHAP `model.py`'ın işidir.

1) İKİ KATMANLI DOĞRULAMA (kritik ayrım)
   Pydantic sınırları EĞİTİM ARALIĞI DEĞİLDİR. İki katman farklı soru sorar:

       Pydantic (burası)  -> "Bu değer fiziksel/mantıksal olarak mümkün mü?"
                             Geniş ama makul sınır. Amaç: çöp veriyi ve
                             negatif/absürt girdiyi modele hiç sokmamak.
       Guardrail (Faz 2)  -> "Model bu değeri eğitimde GÖRDÜ mü?"
                             Dar eğitim aralığı (metadata.training_ranges).

   Pydantic sınırlarını eğitim aralığına eşitlersek guardrail asla
   tetiklenemez ve Faz 2 anlamsızlaşır. Örnek: `witnesses` eğitimde 0-3'tür,
   ama 7 tanıklı bir kaza fiziksel olarak mümkündür — Pydantic kabul eder,
   guardrail "bunu daha önce görmedim" uyarısı verir. Doğru davranış budur.

   Tek istisna: bazı alanların fiziksel sınırı zaten eğitim aralığıyla
   çakışır (ör. `incident_hour_of_the_day` gerçekten 0-23'tür). Bu bilinçli,
   alan bazında yorumlandı.

   FAZ 2 İÇİN NOT — TEK YÖNLÜ KÖRLÜK (reviewer MINOR-7):
   Sekiz alanda Pydantic ALT sınırı eğitim minimumuna EŞİT, yani o alanlarda
   "eğitim aralığının altında" yönünde OOD hiç tetiklenemez:

       months_as_customer (0), capital-gains (0), incident_hour_of_the_day (0),
       number_of_vehicles_involved (1), bodily_injuries (0), witnesses (0),
       injury_claim (0), property_claim (0)

   Üst sınırda aynı durum iki alanda var: `capital-loss` (0) ve yine
   `incident_hour_of_the_day` (23).

   Bu bir hata DEĞİL, fiziksel gerçeğin sonucu: tanık sayısı 0'ın altına,
   araç sayısı 1'in altına inemez; bu veri setinde sermaye zararı pozitif
   olamaz. Eğitim seti bu alanların fiziksel tabanına zaten değiyor.
   Sınırlar BİLEREK değiştirilmedi — gevşetmek fiziksel olarak imkânsız
   girdileri modele sokmak olurdu. Faz 2 guardrail'i "bu alanda alt yönde
   uyarı üretilemiyor" durumunu bir eksiklik sanmamalı.

2) KATEGORİK ALANLAR `Literal` İLE KAPALI
   Pipeline'daki OrdinalEncoder `unknown_value=-1` ile kurulu; bilinmeyen bir
   kategori geldiğinde patlamaz, sessizce -1'e kodlar ve model anlamsız ama
   "başarılı" görünen bir tahmin üretir. Sessiz saçmalık yerine API
   seviyesinde 422 dönmek doğrusu.

   Kategori listeleri `models/metadata.json -> training_ranges` ile birebir
   aynıdır. Elle yazıldılar (OpenAPI şemasında ve /docs'ta okunabilir olsun
   diye), ama `tests/test_api.py::test_literal_choices_match_metadata`
   ikisinin senkron kaldığını her koşuda doğrular.

3) HER ALAN OPSİYONEL
   CLAUDE.md: "Verilmeyen alanlar defaults'taki medyan/mod ile doldurulur."
   34 alanın hepsi `None` varsayılanlıdır. Doldurma İŞLEMİ burada değil,
   `model.py` içinde tek bir yerde yapılır.

4) "?" AYRI BİR ANLAMDIR (alan yok ≠ alan "?")
   `collision_type`, `property_damage`, `police_report_available` alanlarında
   kaynak veri setinde eksiklik `"?"` string'i olarak kodlanmıştır. Bu üç
   alanın `Literal`ı `"?"` değerini de kabul eder:

       alan hiç gönderilmedi (None) -> metadata.defaults'taki mod değeri
       alan "?" gönderildi          -> NaN -> pipeline'ın SimpleImputer'ı

   İkisi bu modelde aynı sonuca varır ama aynı ŞEY değildir: ilki "bilgi
   toplanmadı, varsayılanı kullan", ikincisi "kaynak sistemde bu alan
   bilinmiyor olarak işaretli". Ayrımı korumak, defaults değişirse ya da
   imputer stratejisi değişirse davranışın doğru kalmasını sağlar.

5) `extra="forbid"`
   Bilinmeyen alan sessizce yutulmaz. `capitalgains` yazan bir istemci 422
   alır; aksi hâlde alanı gönderdiğini sanıp aslında varsayılanla skorlanır.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# CLAUDE.md'deki örnek istek — Swagger UI'da "Try it out" için hazır gelsin.
# --------------------------------------------------------------------------- #
PREDICT_REQUEST_EXAMPLE = {
    "incident_severity": "Major Damage",
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Side Collision",
    "auto_year": 2010,
    "number_of_vehicles_involved": 1,
    "witnesses": 2,
    "police_report_available": "YES",
    "capital-gains": 30000,
    "capital-loss": 0,
    "total_claim_amount": 55000,
}


class PredictRequest(BaseModel):
    """`POST /predict` gövdesi — pipeline'ın 34 girdi alanı, hepsi opsiyonel.

    Alan sırası `metadata.feature_list.pipeline_input_order` ile aynı tutuldu
    ki /docs'taki form, modelin gerçek girdi şemasıyla aynı sırada okunsun.

    ÖNEMLİ (K2): `incident_date` / `policy_bind_date` API yüzeyinde YOKTUR.
    Pipeline tarih parse etmez; gerçek girdi şemasında zaten `incident_year` ve
    `policy_bind_year` vardır. Ham tarih kabul edip yıl türetmek, API'ye
    modelin görmediği bir kavram (tarih ayrıştırma, timezone, format hataları)
    sokardı. Doğrudan yıl istiyoruz.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [PREDICT_REQUEST_EXAMPLE]},
    )

    # --- Poliçe / müşteri ------------------------------------------------- #

    # Eğitim: 0-479 ay. Fiziksel tavan olarak 1200 ay (=100 yıl) seçildi;
    # bundan uzun bir müşteri ilişkisi veri giriş hatasıdır.
    months_as_customer: Annotated[int, Field(ge=0, le=1200)] | None = None

    # Eğitim: 20-64. Sınır 16 (birçok eyalette ehliyet alt sınırı) - 120
    # (insan ömrü tavanı). 70 yaşındaki bir sigortalı geçerli bir girdidir;
    # modelin onu görmemiş olması guardrail'in söyleyeceği ayrı bir şeydir.
    age: Annotated[int, Field(ge=16, le=120)] | None = None

    policy_state: Literal["IL", "IN", "OH"] | None = None

    policy_csl: Literal["100/300", "250/500", "500/1000"] | None = None

    # Eğitim: 500-2000. Muafiyet negatif olamaz; 100k makul bir üst tavan.
    policy_deductable: Annotated[float, Field(ge=0, le=100_000)] | None = None

    # Eğitim: 433.33-2047.59. Prim negatif olamaz. Ondalıklı kabul edilir
    # (kaynak kolon zaten float64).
    policy_annual_premium: Annotated[float, Field(ge=0, le=100_000)] | None = None

    # Eğitim aralığı -1.000.000 ile 10.000.000 arası. DİKKAT: eğitim verisinde
    # NEGATİF umbrella_limit var (veri kalitesi sorunu). Alt sınırı 0 yapsaydık
    # modelin eğitimde gördüğü değerleri 422 ile reddederdik — Pydantic sınırı
    # eğitim aralığının ÜST KÜMESİ olmak zorunda.
    umbrella_limit: Annotated[float, Field(ge=-10_000_000, le=100_000_000)] | None = None

    insured_sex: Literal["FEMALE", "MALE"] | None = None

    insured_education_level: (
        Literal["Associate", "College", "High School", "JD", "MD", "Masters", "PhD"] | None
    ) = None

    insured_occupation: (
        Literal[
            "adm-clerical",
            "armed-forces",
            "craft-repair",
            "exec-managerial",
            "farming-fishing",
            "handlers-cleaners",
            "machine-op-inspct",
            "other-service",
            "priv-house-serv",
            "prof-specialty",
            "protective-serv",
            "sales",
            "tech-support",
            "transport-moving",
        ]
        | None
    ) = None

    insured_hobbies: (
        Literal[
            "base-jumping",
            "basketball",
            "board-games",
            "bungie-jumping",
            "camping",
            "chess",
            "cross-fit",
            "dancing",
            "exercise",
            "golf",
            "hiking",
            "kayaking",
            "movies",
            "paintball",
            "polo",
            "reading",
            "skydiving",
            "sleeping",
            "video-games",
            "yachting",
        ]
        | None
    ) = None

    insured_relationship: (
        Literal["husband", "not-in-family", "other-relative", "own-child", "unmarried", "wife"]
        | None
    ) = None

    # Eğitim: 0-100.500. Sermaye KAZANCI tanımı gereği negatif olamaz
    # (zarar ayrı bir kolonda tutuluyor).
    # `alias`: JSON anahtarı tireli (`capital-gains`) çünkü pipeline'ın kolon
    # adı öyle; Python tarafında tire geçerli tanımlayıcı değil.
    capital_gains: Annotated[float, Field(ge=0, le=10_000_000)] | None = Field(
        default=None, alias="capital-gains"
    )

    # Eğitim: -111.100 ile 0 arası. Bu veri setinde zarar NEGATİF işaretle
    # kodlanır; pozitif bir değer işaret hatasıdır ve modele ters yönde
    # sinyal verirdi. Bu yüzden `le=0` bilinçli bir mantıksal kısıttır.
    capital_loss: Annotated[float, Field(ge=-10_000_000, le=0)] | None = Field(
        default=None, alias="capital-loss"
    )

    # --- Olay ------------------------------------------------------------- #

    incident_type: (
        Literal["Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"]
        | None
    ) = None

    # "?" = kaynak sistemde bilinmiyor -> NaN -> pipeline imputer'ı (bkz. 4).
    collision_type: (
        Literal["Front Collision", "Rear Collision", "Side Collision", "?"] | None
    ) = None

    incident_severity: (
        Literal["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"] | None
    ) = None

    authorities_contacted: Literal["Ambulance", "Fire", "Other", "Police"] | None = None

    incident_state: Literal["NC", "NY", "OH", "PA", "SC", "VA", "WV"] | None = None

    incident_city: (
        Literal[
            "Arlington",
            "Columbus",
            "Hillsdale",
            "Northbend",
            "Northbrook",
            "Riverwood",
            "Springfield",
        ]
        | None
    ) = None

    # İSTİSNA: burada fiziksel sınır eğitim aralığıyla (0-23) aynıdır, çünkü
    # bir günün saati tanımı gereği 0-23'tür. Bu alanda sayısal OOD uyarısı
    # hiç tetiklenemez; bu bir eksiklik değil, alanın doğası.
    incident_hour_of_the_day: Annotated[int, Field(ge=0, le=23)] | None = None

    # Eğitim: 1-4. Zincirleme kazada çok daha fazlası mümkün; tavan 100.
    number_of_vehicles_involved: Annotated[int, Field(ge=1, le=100)] | None = None

    property_damage: Literal["NO", "YES", "?"] | None = None

    # Eğitim: 0-2. Çok yaralılı kaza mümkün; tavan 20.
    bodily_injuries: Annotated[int, Field(ge=0, le=20)] | None = None

    # Eğitim: 0-3. 10'a kadar tanık makul; guardrail 3'ün üstünü uyarır.
    witnesses: Annotated[int, Field(ge=0, le=10)] | None = None

    police_report_available: Literal["NO", "YES", "?"] | None = None

    # --- Tazminat kalemleri ------------------------------------------------ #
    # Hepsi negatif olamaz (ödenen tutar). Tavan 10.000.000: eğitimdeki en
    # yüksek talep ~115k, ama büyük ticari hasarlar bu mertebeye çıkabilir.
    # Tutarlar float kabul edilir (kuruş/cent'li tutar gerçekçidir).

    total_claim_amount: Annotated[float, Field(ge=0, le=10_000_000)] | None = None
    injury_claim: Annotated[float, Field(ge=0, le=10_000_000)] | None = None
    property_claim: Annotated[float, Field(ge=0, le=10_000_000)] | None = None
    vehicle_claim: Annotated[float, Field(ge=0, le=10_000_000)] | None = None

    # --- Araç / tarih türevleri -------------------------------------------- #

    auto_make: (
        Literal[
            "Accura",
            "Audi",
            "BMW",
            "Chevrolet",
            "Dodge",
            "Ford",
            "Honda",
            "Jeep",
            "Mercedes",
            "Nissan",
            "Saab",
            "Suburu",
            "Toyota",
            "Volkswagen",
        ]
        | None
    ) = None

    # Yıl alanlarının hepsinde 1900-2100: otomobilin/poliçenin var olabileceği
    # makul takvim penceresi. Eğitim aralıkları çok daha dar
    # (auto_year 1995-2015, policy_bind_year 1990-2015, incident_year sabit
    # 2015) — aradaki fark bilerek bırakıldı, guardrail'in işi.
    auto_year: Annotated[int, Field(ge=1900, le=2100)] | None = None
    incident_year: Annotated[int, Field(ge=1900, le=2100)] | None = None
    policy_bind_year: Annotated[int, Field(ge=1900, le=2100)] | None = None


# --------------------------------------------------------------------------- #
# /predict yanıtı
# --------------------------------------------------------------------------- #


class ShapValue(BaseModel):
    """Tek bir feature'ın SHAP katkısı — CLAUDE.md sözleşmesiyle birebir.

    `value` ve `base_value` LOG-ODDS (raw margin) uzayındadır, olasılık değil.
    Toplamları modelin ham skorunu verir: sum(value) + base_value = raw margin,
    sigmoid(raw margin) = fraud_probability. Frontend'in waterfall grafiği bu
    toplanabilirlik sayesinde matematiksel olarak doğru çizilebilir.
    """

    feature: str = Field(description="Temiz feature adı (cat__/remainder__ öneki yok).")
    value: float = Field(description="Bu feature'ın log-odds katkısı.")
    base_value: float = Field(description="Modelin taban log-odds değeri (her elemanda aynı).")


class PredictResponse(BaseModel):
    """`POST /predict` yanıtı — CLAUDE.md sözleşmesindeki 4 alan, fazlası yok.

    Bilinçli olarak EKSİK olanlar: girdinin yankılanması (echo), model
    versiyonu, dahili id'ler. Yanıt ne kadar dar olursa sızıntı yüzeyi o kadar
    küçük olur; sözleşmede olmayan hiçbir alan eklenmez.
    """

    fraud_probability: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    shap_values: list[ShapValue] = Field(
        description="34 feature, abs(value) azalan sırada. Kaçının gösterileceğine frontend karar verir."
    )
    # FAZ 2 İÇİN KRİTİK NOT — GUARDRAIL YALNIZCA SAYISAL OOD ÜRETEBİLİR
    # (Codex C-4). Kategorik alanlar `Literal` ile eğitimde görülen değerlere
    # kapalı: bilinmeyen bir kategori guardrail'e HİÇ ULAŞMADAN 422 alır. Yani
    # "eğitim dışı `policy_state` gönderdim, uyarı bekliyorum" senaryosu
    # imkânsızdır — 422 döner.
    #
    # Bu bilinçli ve doğru: `OrdinalEncoder` `unknown_value=-1` ile kurulu,
    # bilinmeyen kategoriyi sessizce -1'e kodlayıp anlamsız ama "başarılı"
    # görünen bir tahmin üretirdi (bkz. bu dosyanın girişindeki 2. madde).
    # Ama alanın adı ve açıklaması bu sınırı gizlememeli: Faz 4'teki UI
    # banner'ı "kategorik OOD uyarısı gelir" varsayımıyla yazılırsa sessizce
    # yanlış olur.
    out_of_distribution_warnings: list[str] = Field(
        description=(
            "Eğitim aralığının dışında kalan SAYISAL alan adları. Kategorik "
            "alanlar API şemasında kapalı olduğundan burada görünmez — geçersiz "
            "bir kategori uyarı değil 422 üretir. Faz 1'de her zaman boş döner; "
            "guardrails.py (Faz 2) dolduracak."
        )
    )


# --------------------------------------------------------------------------- #
# /health yanıtı
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    """Basit healthcheck.

    `protected_namespaces=()`: Pydantic v2 varsayılan olarak `model_` ile
    başlayan alan adlarını kendi API'siyle çakışma riski diye uyarır. Bizim
    alan adlarımız (`model_loaded`, `model_version`) metadata sözleşmesinden
    geliyor; uyarıyı susturmak yerine namespace korumasını kapatıyoruz.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    # Sadece "süreç ayakta" değil "artefakt gerçekten yüklendi" bilgisi.
    model_loaded: bool
    # Deploy sonrası "hangi artefakt koşuyor?" sorusunu tek istekle cevaplar.
    model_version: str


# --------------------------------------------------------------------------- #
# /model-info yanıtı — BEYAZ LİSTE PROJEKSİYONU
# --------------------------------------------------------------------------- #
#
# metadata.json OLDUĞU GİBİ DÖNDÜRÜLMEZ. Aşağıdaki modeller açık bir beyaz
# liste kurar: her alt model yalnızca izin verilen alanları ilan eder ve
# `extra="ignore"` sayesinde metadata'daki diğer her şey sessizce düşer.
#
# Neden beyaz liste, kara liste değil: yarın train_pipeline.py metadata'ya yeni
# bir alan eklerse (ör. bir iç maliyet metriği), kara listede o alan otomatik
# olarak public olurdu. Beyaz listede ise açıkça eklenene kadar dışarı çıkmaz.
#
# Dışarıda bırakılanlar ve gerekçeleri:
#   preprocessing_contract -> implementasyon detayı; istemcinin işine yaramaz,
#                             sadece iç mimariyi ifşa eder.
#   model_params           -> hiperparametreler (n_estimators, learning_rate...).
#   dataset.source_file    -> sunucu tarafındaki dosya adı; gereksiz.


class MetricsInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_pr_auc: float
    test_pr_auc: float
    metric_name: str


# DİKKAT — AŞAĞIDAKİ AÇIKLAMALAR BİLİNÇLİ OLARAK DOCSTRING DEĞİL, YORUM:
# Pydantic model docstring'leri OpenAPI şemasına `description` olarak girer ve
# `/openapi.json` public bir yüzeydir. Beyaz listenin DIŞINDA bıraktığımız
# metadata anahtarlarının adlarını docstring'de anmak, `/model-info`'dan
# özenle çıkardığımız iç yapıyı arka kapıdan public şemaya taşırdı.
# Kaynağı okuyan için eğitici değer aynı; OpenAPI için görünmez.
#
#   DatasetInfo      -> metadata'daki kaynak dosya adı alanı BİLEREK yok;
#                       sunucu tarafındaki dosya adı istemcinin işine yaramaz.
#   FeatureListInfo  -> SHAP eşlemesi için kullanılan dönüştürülmüş kolon adı
#                       listeleri beyaz listede YOK: onlar iç detaydır ve
#                       `/predict` yanıtında zaten temiz adla görünürler.
class DatasetInfo(BaseModel):
    """Eğitim veri kümesinin boyut ve split bilgisi."""

    model_config = ConfigDict(extra="ignore")

    n_rows: int
    n_train: int
    n_test: int
    test_size: float
    random_state: int
    positive_rate_train: float
    positive_rate_test: float


class FeatureListInfo(BaseModel):
    """Model card'ın feature bölümü."""

    model_config = ConfigDict(extra="ignore")

    pipeline_input_order: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    # NOT: burada PII kolon ADLARI listelenir (policy_number, insured_zip,
    # incident_location). Bu bir sızıntı DEĞİL, tersi: "bu alanlar modele hiç
    # girmedi" beyanıdır. Sızıntı, PII *değerlerinin* dönmesi olurdu — model
    # bu kolonları hiç görmediği için mümkün de değildir.
    dropped_columns: list[str]
    target: str


class TrainingRangeInfo(BaseModel):
    """Guardrail'in (Faz 2) dayanacağı "eğitimde ne gördük" kaydı.

    Sayısal alanlarda min/max, kategorik alanlarda categories dolu gelir;
    diğeri None kalır. `int | float` birleşimi tamsayı sınırların yanıtta
    2015.0 gibi görünmesini engeller.
    """

    model_config = ConfigDict(extra="ignore")

    type: Literal["numeric", "categorical"]
    dtype: str
    min: int | float | None = None
    max: int | float | None = None
    categories: list[str] | None = None


class DefaultInfo(BaseModel):
    """Verilmeyen alanlar için kullanılan medyan/mod değeri.

    Frontend bunu form placeholder'ı olarak gösterebilsin diye public: kullanıcı
    "boş bıraktığım alan neyle dolduruldu?" sorusunun cevabını görebilmeli.
    """

    model_config = ConfigDict(extra="ignore")

    value: int | float | str
    dtype: str
    strategy: Literal["median", "mode"]


class FeatureInfluenceEntry(BaseModel):
    """Tek bir feature'ın eğitilmiş modeldeki ÖLÇÜLEN etkisi."""

    model_config = ConfigDict(extra="ignore")

    split_count: int = Field(description="Ağaçlarda bu kolon üzerinde yapılan bölünme sayısı.")
    gain: float = Field(description="Bu bölünmelerin toplam kazancı.")
    has_influence: bool = Field(description="split_count > 0 — model bu feature'ı kullandı mı?")


class FeatureInfluenceSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_features: int
    n_with_influence: int
    n_without_influence: int
    features_without_influence: list[str]


class FeatureInfluenceInfo(BaseModel):
    """Hangi feature'ın tahmini gerçekten etkilediğinin ölçümü.

    Faz 3'teki formun "bu alan modeli etkilemiyor" rozetini besler. Frontend'in
    bu listeyi kendi içine gömmesi YASAK: model yeniden eğitilince liste
    değişir ve gömülü kopya sessizce yalan söylemeye başlar. Tek doğruluk
    kaynağı artefakt, tek dağıtım kanalı bu endpoint.

    Anahtarlar API alan adlarıyla birebir aynıdır (`capital-gains` tireli
    hâliyle), yani frontend ayrıca bir eşleme tablosu tutmaz.
    """

    model_config = ConfigDict(extra="ignore")

    description: str
    measured_from: str
    interpretation_note: str
    summary: FeatureInfluenceSummary
    features: dict[str, FeatureInfluenceEntry]


class FairnessAttributeInfo(BaseModel):
    """Korunan/vekil nitelik — BEYAN ve ÖLÇÜM yan yana.

    `used_as_model_feature` bir beyandır: feature modele verildi mi?
    `has_influence` bir ölçümdür: eğitilmiş ağaçlar onu gerçekten kullandı mı?
    İkisi farklı şeylerdir ve model card ikisini de göstermek zorundadır —
    yalnızca beyanı göstermek okuyucuyu yanlış yönlendirir.
    """

    model_config = ConfigDict(extra="ignore")

    feature: str
    basis: str
    rationale: str
    # BEYAN: feature modele girdi olarak verildi.
    used_as_model_feature: bool
    # ÖLÇÜM: booster'dan hesaplanır, elle yazılmaz.
    split_count: int
    has_influence: bool
    # Yalnızca `protected_attributes_used_as_features` girdilerinde var.
    severity: str | None = None


class FairnessInfo(BaseModel):
    """Model card'ın fairness bölümü — Faz 5 bunu tabloya basacak.

    Beyaz listede TUTULUYOR (K9). Modelin korunan nitelikleri feature olarak
    kullandığını saklamak, bir demo için bile yanlış olurdu.
    """

    model_config = ConfigDict(extra="ignore")

    status: str
    # "beyan" ile "ölçüm" alanlarının farkını okuyucuya anlatan metin;
    # model card sayfası bunu olduğu gibi basacak.
    field_semantics: str
    protected_attributes_used_as_features: list[FairnessAttributeInfo]
    proxy_risk_attributes: list[FairnessAttributeInfo]
    audit_performed: bool
    audit_metrics_computed: list[str]
    intended_use: str
    production_requirements: list[str]
    notes: str


class RiskThresholdsInfo(BaseModel):
    """`risk_level` etiketinin nasıl üretildiğinin açık beyanı.

    Frontend eşikleri kendi kopyalamasın diye buradan okur; tek doğruluk
    kaynağı `model.py`'daki sabitlerdir.
    """

    model_config = ConfigDict(extra="ignore")

    low_below: float
    high_at_or_above: float
    rule: str


class CalibrationInfo(BaseModel):
    """Olasılıkların kalibre OLMADIĞININ açık uyarısı.

    Bunu model card'da söylemek zorunludur: model dengesiz sınıf ağırlığıyla
    eğitildi, dolayısıyla `fraud_probability` "100 vakadan 71'i dolandırıcı"
    anlamına GELMEZ. Sıralama (ranking) için güvenilir, mutlak olasılık olarak
    değil. Bu uyarı olmadan bir sigorta müşterisi sayıyı yanlış okur.
    """

    model_config = ConfigDict(extra="ignore")

    calibrated: bool
    method: str
    warning: str


class ModelInfoResponse(BaseModel):
    """`GET /model-info` yanıtı — model card sayfasının (Faz 5) veri kaynağı."""

    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    model_name: str
    model_version: str
    algorithm: str
    trained_at: str
    metrics: MetricsInfo
    dataset: DatasetInfo
    feature_list: FeatureListInfo
    feature_influence: FeatureInfluenceInfo
    training_ranges: dict[str, TrainingRangeInfo]
    defaults: dict[str, DefaultInfo]
    fairness: FairnessInfo
    library_versions: dict[str, str]
    risk_thresholds: RiskThresholdsInfo
    probability_calibration: CalibrationInfo
