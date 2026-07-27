# Faz 1 — FastAPI Endpoint'leri + Pydantic Validation · Review Log

| | |
|---|---|
| **Tarih** | 2026-07-27 |
| **Branch** | `feat/phase-1-fastapi-api` |
| **Faz** | 1 — FastAPI endpoint'leri + Pydantic validation |
| **Bitti Kriteri (CLAUDE.md)** | `/predict`, `/health`, `/model-info` çalışıyor, pytest yeşil |
| **Reviewer** | `reviewer` agent (Adım 1 iç kontrol, bağımsız koşuldu) |
| **İkinci görüş** | Codex CLI 0.145.0 · `--sandbox read-only` |
| **İncelenen dosyalar** | `backend/app/{__init__,main,model,schemas}.py`, `backend/tests/{__init__,conftest,test_api}.py`, `backend/ruff.toml`, `backend/requirements.txt` |
| **Adım 1 sonucu** | **PASS** (Bitti kriteri sağlanıyor) |

---

## Adım 1 — İç Kontrol

| # | Kontrol | Sonuç |
|---|---|---|
| 1 | `pytest` (backend/) | **PASS** — 58 passed, 4 warnings, 3.74s |
| 2 | `pytest` (repo kökünden) | **PASS** — 58 passed (rootdir çıkarımı çalışıyor) |
| 3 | `ruff check .` | **PASS** — `All checks passed!` (uvx ruff 0.14.5) |
| 4 | `mypy` | **KOŞULMADI** — ortamda kurulu değil, projede yapılandırma yok |
| 5 | `ruff format --check` (bilgi) | 3 inceleme dosyası + `train_pipeline.py` yeniden biçimlendirilirdi (format zorunlu değil, `check` yeşil) |
| 6 | Test sırası bağımsızlığı | **PASS** — 58 test TERS sırada da yeşil; fail-fast testleri tek başına da yeşil |
| 7 | GERÇEK uvicorn + gerçek HTTP (TestClient değil) | **PASS** — aşağıda |
| 8 | Mutasyon testi (24 mutasyon) | **20 öldürüldü / 4 hayatta kaldı** (%83) — aşağıda |
| 9 | Eş zamanlılık (300 HTTP + 1280 doğrudan çağrı) | **PASS** — 0 tutarsızlık |
| 10 | Artefakt bütünlüğü (tüm testlerden sonra) | **PASS** — `pipeline.pkl` / `metadata.json` sha256 değişmedi |

### 1. Bitti kriteri — GERÇEK sunucu, GERÇEK HTTP

`python -m uvicorn app.main:app --host 127.0.0.1 --port 8731` ile ayağa kaldırıldı,
istekler `curl` ile atıldı. TestClient'a güvenilmedi.

```
GET  /health     -> 200  {"status":"ok","model_loaded":true,"model_version":"1.0.0"}
GET  /model-info -> 200  13 üst seviye anahtar, test_pr_auc=0.6527088560280899
POST /predict    -> 200  fraud_probability=0.7299506078527795  risk_level="high"
GET  /docs       -> 200      GET /redoc -> 200      GET /openapi.json -> 200
DELETE /health   -> 405  (allow_methods=["GET","POST"] gerçekten uygulanıyor)
```

`/predict` gecikmesi: ~40-47 ms (5 ardışık istek). Üç endpoint de çalışıyor →
**bitti kriteri SAĞLANIYOR.**

### 2. API sözleşmesi uyumu (CLAUDE.md satır 84-91)

Ham yanıt metni üzerinden doğrulandı:

```
TOP-LEVEL KEYS: ['fraud_probability', 'out_of_distribution_warnings', 'risk_level', 'shap_values']
shap item keys: {('base_value', 'feature', 'value')}   (34 elemanın hepsinde aynı)
n_shap: 34    distinct base_value: {-0.8642788220922653}   ood: []
sigmoid(sum(shap)+base) = 0.7299506078527803  vs  fraud_probability = 0.7299506078527795
```

**Fazla alan yok, eksik alan yok.** SHAP toplanabilirliği ~1e-15 hassasiyetle
sağlanıyor → pozitif sınıf ekseni seçimi matematiksel olarak doğru.

> **DOKÜMAN TUTARSIZLIĞI (CLAUDE.md'nin kendisinde):** CLAUDE.md satır 21 alan adını
> `shap_value`, satır 88'deki API sözleşmesi örneği ise `value` diyor. Implementasyon
> `value` kullanıyor — yani açık sözleşme bloğuna uyuyor, doğrusu bu. Ama Faz 4
> (frontend recharts) başlamadan CLAUDE.md satır 21 düzeltilmeli.

### 3. PII sızıntısı — ham JSON metin taraması

`/predict` (örnek payload + boş gövde) ve `/model-info` yanıtlarının **ham metni** tarandı:

| Anahtar | `/predict` | `/model-info` |
|---|---|---|
| `policy_number`, `insured_zip`, `incident_location` | yok | sadece `feature_list.dropped_columns` içinde |
| `auto_model`, `incident_date`, `policy_bind_date`, `_c39` | yok | sadece `dropped_columns` içinde |
| `preprocessing_contract`, `model_params`, `source_file` | yok | yok |
| `n_estimators`, `learning_rate`, `num_leaves` | yok | yok |
| dosya yolu / kullanıcı adı / `.pkl` | yok | yok |

`dropped_columns` çıkarıldığında geriye kalan JSON'da bu adların **hiçbiri**
geçmiyor. PII *değeri* hiçbir yerde dönmüyor.

### 4. "Elle encoding sözlüğü YOK" kuralı

`app/` altında kategori→sayı dönüşümü yapan **tek** kod `preprocessor.transform()`
çağrısı (`model.py:367`) — yani pipeline'ın kendi transformer'ı. `map()`, `replace()`,
`LabelEncoder`, `get_dummies`, elle sözlük: **yok**. `prepare_row()` sadece şema
normalizasyonu yapıyor (default doldurma, `"?"`→NaN, kolon sıralama).
`test_api.py:314`'teki `encoder.categories_[...].index(...)` bir *test* hesabıdır ve
değeri **fitted artefakttan** okur, sabit kodlamaz. **Kural korunuyor.**

### 5. Eş zamanlılık — GERÇEKTEN test edildi

**Test A — 300 paralel HTTP isteği** (20 worker × 15 tur, 8 ayrışan payload, sıra
kasten kırıldı). Her yanıt kendi seri referansıyla bit bazında karşılaştırıldı.
→ **UYUŞMAYAN: 0**

**Test B — 1280 doğrudan `bundle.predict()` çağrısı** (32 thread, `threading.Barrier`
ile ilk dalga senkron başlatıldı, HTTP gecikmesi devre dışı → maksimum çakışma).
→ **UYUŞMAYAN: 0**

Paylaşılan durum mutasyonu iddiası **DOĞRULANDI**: `explainer.expected_value` her
çağrıda yeni bir nesneye atanıyor (id her seferinde farklı), ama değer sabit
(`-0.8642788220922653`), dolayısıyla yarış zararsız. `predict_proba`'nın kilitsiz
olması bu kurulumda sorun ÜRETMİYOR. Kilit gereksiz ama zararsız.

**Ancak:** kilidi kaldıran mutasyon (M8) test paketini yeşil bıraktı → eş zamanlılık
davranışının **hiçbir regresyon testi yok** (bkz. MINOR-9).

### 6. Fail-fast — dosya GERÇEKTEN yeniden adlandırılarak test edildi

Monkeypatch değil. `models/pipeline.pkl` diskte yeniden adlandırıldı, gerçek uvicorn
süreci başlatıldı:

```
pipeline.pkl YOKKEN:   port açılmadı (served=False), süreç çıkış kodu = 3
                       Traceback: main.py:86 lifespan -> model.py:243 raise ArtifactError
pipeline.pkl VARKEN:   "Application startup complete." — port açıldı
```

**Fail-fast iddiası GERÇEK.** Geri alma doğrulaması:

```
pipeline.pkl  sha256 = ca80b84f4a994fba4faa3d7f54a2521593077f4dd5e8177b267e969a0c8f7abe  (BASELINE ile AYNI)
metadata.json sha256 = 280049df6783aa4c7d04e45eaf10a4914aae3747ac031624027d37a749ff1658  (BASELINE ile AYNI)
geçici dosya kaldı mı: False
```

### 7. Mutasyon testi — testler gerçekten bir şey doğruluyor mu?

Kaynak dosyalar git'te izlenmediği için önce `cp` ile yedek alındı; her mutasyon
`try/finally` içinde uygulandı ve yedekten **bayt bazında** geri yazıldı, ardından
sha256 doğrulandı.

**ÖLDÜRÜLEN 20 mutasyon (test kırmızıya döndü):**

| # | Mutasyon | Yakalayan test |
|---|---|---|
| M1 | `classify_risk` high eşiği `>=` → `>` | `test_risk_level_thresholds` |
| M2c | `"?"` → NaN dönüşümü atlandı | `test_question_mark_becomes_nan_not_an_unknown_category` |
| M3 | SHAP `abs(value)` sıralaması kaldırıldı | `test_shap_output_shape_and_naming` |
| M4 | `order_columns` adımı bozuldu | `test_prepare_row_applies_order_columns_step` |
| M5 | `/model-info`'ya `preprocessing_contract` eklendi | `test_model_info_does_not_leak_internal_keys` |
| M6 | `PredictResponse`'a echo alanı eklendi | `test_predict_response_has_exactly_the_contract_fields` + PII testi |
| M7 | `base_value` 0'a sabitlendi | 48 error (load-time additivity) |
| M9 | defaults yerine sabit 0 dolduruldu | `test_missing_field_falls_back_to_metadata_default` |
| M10 | CORS `allow_origins=["*"]` | `test_cors_allows_vite_origin_and_is_not_wildcard` |
| M11b | Artefakt fail-fast kaldırıldı | `test_app_refuses_to_start_without_artifact`, `test_missing_artifact_fails_fast` |
| M12 | `capital-loss` `le=0` → `le=10_000_000` | `test_predict_rejects_invalid_input` |
| M13 | `witnesses` üst sınırı eğitim maks.'a çekildi | `test_numeric_bounds_are_supersets_of_training_ranges` |
| M14 | `policy_state` Literal'ine `"TX"` eklendi | `test_literal_choices_match_metadata` |
| M15 | `extra="forbid"` → `"ignore"` | `test_predict_rejects_invalid_input` (6 param) |
| M16 | `bodily_injuries` alanı silindi | `test_request_schema_covers_every_pipeline_input` |
| M17 | SHAP değerleri işaret ters çevrildi | 48 error (load-time additivity) |
| M18 | SHAP feature adına PII adı yazıldı | `test_predict_never_leaks_pii_column_names` |
| M19 | `risk_level` hep `"low"` döndü | `test_predict_with_claude_md_example` |
| M20 | `transformed_display_names` → `transformed_order` (ham `cat__` önekleri) | 48 error (load-time) |
| M22 | `ALLOWED_ORIGINS` ortam değişkeni adı yanlış yazıldı | `test_cors_env_var_is_parsed_as_comma_separated_list` |

**HAYATTA KALAN 4 mutasyon (test YAKALAYAMADI):**

| # | Mutasyon | Anlamı |
|---|---|---|
| **M21** | `allow_origins=get_allowed_origins()` → `["http://localhost:5173"]` sabit kodlandı | **Ortam değişkeni → middleware kablolaması HİÇ test edilmiyor** (bkz. MAJOR-1) |
| **M23** | `allow_methods=["GET","POST"]` → `["*"]` | K10 iddiası sadece origin için doğrulanıyor |
| **M24** | `allow_credentials=True` eklendi | Docstring'in açık iddiası test edilmiyor |
| M8 | `_shap_lock` kaldırıldı | Eş zamanlılık regresyon testi yok |

**Mutasyon skoru: 20/24 = %83.** Sonuç: testler "yeşil ama boş" DEĞİL — sözleşme,
sızıntı, imputasyon, SHAP ekseni ve doğrulama sınırları gerçekten korunuyor. Tek
kör nokta **CORS middleware kablolaması**.

#### Geri yükleme kanıtı (mutasyon sonrası)

```
=== SON GERİ YÜKLEME DOĞRULAMASI ===
  geri yükleme sonrası pytest: rc=0 -> 58 passed, 4 warnings in 3.48s
  sha256 tam eşleşme: True

app/__init__.py    8a0115687be614912d7359a3f49598f18292535f419541076e4b85545cb5aa06
app/main.py        833f59139676b4ddd121498c960b614096c992a38f266f90d689d59cdb2d4eff
app/model.py       dae1d058b9c6455c232f2ef7864179bca47cc4498a5355719c4b2ec9ca320a46
app/schemas.py     62c02742052808df1bea0f4bf9ca710bf98342c9b11fa330a1710a5a85f983cd
tests/__init__.py  d82e91f42fd1b9b01a5a5a013bdc245c38b8ac78310023e101ba1e994312ffe1
tests/conftest.py  9abc0e8bd631ef608c41b87a54cd7f27cfeaacd12faa215feed10cd78d02f98a
tests/test_api.py  e9ecf98f091de79b6058142c4bbcaa91d9f53032e00c24aa24607076f443eaba
```

Dört mutasyon turunun (24 mutasyon) sonunda **tüm dosyalar baseline sha256 ile
birebir eşleşiyor** ve pytest yeşil.

### 8. `get_allowed_origins()` import zamanında bir kez çağrılıyor — sorun mu?

**Mekanizma çalışıyor.** `ALLOWED_ORIGINS="https://demo.netlify.app,https://ikinci.example"`
ile gerçek sunucu kaldırıldı:

```
Origin: https://demo.netlify.app  -> 200, access-control-allow-origin: https://demo.netlify.app
Origin: https://ikinci.example    -> 200, access-control-allow-origin: https://ikinci.example
Origin: http://localhost:5173     -> 400 (varsayılan artık geçerli değil — doğru davranış)
```

**Değerlendirme: import-zamanı okuma KENDİ BAŞINA sorun DEĞİL.** HF Spaces'te ortam
değişkeni süreç başlamadan önce set edilir; `app` zaten modül düzeyinde bir singleton
ve CORS politikasının istek başına değişmesi istenmez. Ortam değişkeni değiştiğinde
Space'i yeniden başlatmak zaten gereklidir.

**Ama iki gerçek sonucu var:**

1. **Test edilebilirlik kaybı → gerçek bir kör nokta.** `add_middleware` import
   sırasında çalıştığı için hiçbir test `ALLOWED_ORIGINS`'i set edip middleware
   davranışını uçtan uca doğrulayamıyor. M21 mutasyonu (env'i tamamen yok say,
   origin'i sabit kodla) **58 testin hepsini yeşil bıraktı**. Faz 7'nin bütün
   bahsi tam olarak bu yol.
2. **Sessiz yanlış yapılandırma.** `ALLOWED_ORIGINS=""` set edilirse liste boş olur,
   hiçbir origin'e izin verilmez, uygulama sağlıklı görünür ama tarayıcıdan gelen
   her istek engellenir. Açılışta ne hata ne uyarı var — projenin kendi fail-fast
   felsefesiyle çelişiyor.

### 9. Backend-agent'ın raporladığı noktalar — bağımsız değerlendirmem

#### (a) K9/K11 çelişkisi — `dropped_columns` PII **adlarını** içeriyor

**AGENT HAKLI. Gerçek bir sızıntı riski YOK. Karar doğru, değiştirilmemeli.**

Gerekçe (varsayım değil, ölçüm):
- `/model-info` ham metninde bu üç ad **yalnızca** `feature_list.dropped_columns`
  içinde geçiyor; `dropped_columns` çıkarıldığında kalan JSON'da hiçbiri yok.
- `defaults`, `training_ranges`, `pipeline_input_order` içinde yok.
- Model bu kolonları **hiç görmedi** — booster'ın 34 feature'ı arasında yoklar,
  dolayısıyla PII *değeri* üretmesi teknik olarak imkânsız.
- Açığa çıkan tek bilgi "kaynak şemada `insured_zip` adlı bir kolon vardı" — bu
  zaten herkese açık bir Kaggle veri setinin şeması.

Model card açısından "hangi kolonlar bilerek dışlandı" beyanı **artı değer**;
gizlemek şeffaflık kaybı olurdu. `test_model_info_pii_appears_only_as_dropped_column_declaration`
bu kararı doğru şekilde çitliyor (M18 mutasyonu ile öldürücülüğü kanıtlandı).

#### (b) 34 feature'ın 16'sının LightGBM split sayısı sıfır

**DOĞRULANDI.** `model.booster_.feature_importance(importance_type="split")`.
`booster.feature_name() == metadata.transformed_order` → `True` (eşleme güvenli).

**Split = 0 olan 16 feature:**
`authorities_contacted`, `bodily_injuries`, `collision_type`, `incident_hour_of_the_day`,
`incident_state`, `incident_type`, `incident_year`, `insured_relationship`, `insured_sex`,
`number_of_vehicles_involved`, `police_report_available`, `policy_deductable`,
`policy_state`, `property_damage`, `umbrella_limit`, `vehicle_claim`

**Kullanılan 18 feature (split sayısı):** `insured_hobbies` 353 · `incident_severity` 173 ·
`policy_annual_premium` 107 · `capital-gains` 75 · `auto_year` 74 · `capital-loss` 67 ·
`policy_bind_year` 46 · `witnesses` 37 · `incident_city` 15 · `auto_make` 14 ·
`insured_education_level` 12 · `insured_occupation` 10 · `policy_csl` 8 · `age` 8 ·
`months_as_customer` 7 · `injury_claim` 7 · `property_claim` 5 · `total_claim_amount` 1

**CLAUDE.md örnek isteğinin 10 alanından 4'ü tamamen ETKİSİZ:**

| Alan | Durum |
|---|---|
| `incident_severity` | etkili (split=173, gain=14811.6) |
| `capital-gains` | etkili (75) |
| `auto_year` | etkili (74) |
| `capital-loss` | etkili (67) |
| `witnesses` | etkili (37) |
| `total_claim_amount` | **teknik olarak etkili ama ihmal edilebilir** (split=1, gain=1.2) |
| `incident_type` | **ETKİSİZ (split=0)** |
| `collision_type` | **ETKİSİZ (split=0)** |
| `number_of_vehicles_involved` | **ETKİSİZ (split=0)** |
| `police_report_available` | **ETKİSİZ (split=0)** |

Doğrudan sonucu: `/predict` yanıtındaki 34 SHAP değerinin **16'sı her zaman tam `0.0`**
(ölçüldü). Faz 4'te waterfall grafiği bu 16 satırı çizerse yarısı sıfır uzunluğunda
çubuk olur. Faz 3'teki form da müşteriye "bu alanı değiştir" diyecek ama çıktı
kıpırdamayacak. Bu Faz 1 hatası değil, **Faz 3-4 başlamadan verilmesi gereken bir
ürün kararı** (bkz. MAJOR-3).

#### (c) `question_mark_to_nan` encoder seviyesinde test — yeterli mi?

**YETERLİ. Agent'ın gerekçesi doğru ve kanıtlandı.**

Uçtan uca testin bu adımı yakalayamayacağı iddiası doğrulandı: `collision_type`,
`property_damage`, `police_report_available` üçünün de split sayısı 0 (yukarıda),
dolayısıyla `"?"` yanlışlıkla -1'e kodlansa bile `fraud_probability` değişmez.
Ölçümle teyit: `{"collision_type":"?","property_damage":"?","police_report_available":"?"}`
payload'ı ile `{}` payload'ı **aynı olasılığı** veriyor (0.241139351178779).

Encoder seviyesindeki testin gerçekten öldürücü olduğu mutasyonla kanıtlandı:
M2c (`"?"`→NaN dönüşümünü kaldır) → `test_question_mark_becomes_nan_not_an_unknown_category`
**KIRMIZI**. Test hem `-1` olmadığını hem de imputer'ın mod kodunun geldiğini kontrol
ediyor; ikisi birden doğru soruyu soruyor.

#### (d) Pydantic sınırları eğitim aralığının üst kümesi mi?

**EVET — 18 sayısal alanın 18'i de üst küme. Ters olan TEK BİR alan yok.**
16 kategorik alanın `Literal` seçenekleri de metadata kategorileriyle birebir aynı.

**Ama iki nüans agent'ın raporunda yok:**

**(d-1) Tek taraflı çakışma — 8 alanda ALT sınırda guardrail için yer yok:**
`months_as_customer` (0=0), `capital-gains` (0=0), `incident_hour_of_the_day` (0=0),
`number_of_vehicles_involved` (1=1), `bodily_injuries` (0=0), `witnesses` (0=0),
`injury_claim` (0=0), `property_claim` (0=0).
ÜST sınırda yer olmayan 2 alan: `capital-loss` (0=0), `incident_hour_of_the_day` (23=23).

Bunların çoğu doğal taban (adet/tutar negatif olamaz) — kabul edilebilir. Ancak
`schemas.py:22-25`'teki "Tek istisna: `incident_hour_of_the_day`" ifadesi ve
`test_api.py:479-511`'in `(low, high) != (train_min, train_max)` kontrolü **çifti**
karşılaştırdığı için tek taraflı çakışmayı görmüyor. Belge/test biraz iyimser.

**(d-2) Kategorik alanlarda guardrail TANIM GEREĞİ hiç tetiklenemez.**
`Literal` == eğitim kategorileri olduğu için eğitimde görülmemiş bir kategori
API'den zaten **geçemiyor** (422). Yani `out_of_distribution_warnings` listesinde
hiçbir zaman kategorik alan adı olamaz. CLAUDE.md guardrail'i "min/max dışı" diye
tanımladığı için sözleşmeye aykırı değil, ama Faz 2'nin gerçek kapsamı
**16 kategorik alan hariç, 18 sayısal alan** (8'inde yalnızca üst sınır) demek.

---

## Adım 1 — Reviewer'ın kendi bulguları (önem sırasına göre)

### BLOCKER
Yok. Faz 1 bitti kriteri sağlanıyor, Faz 2'ye geçilebilir.

### MAJOR

**MAJOR-1 — CORS ortam değişkeni → middleware kablolaması test edilmiyor (Faz 7 riski)**
`app/main.py:104-112`. M21 mutasyonu (`allow_origins` sabit kodlanıp
`get_allowed_origins()` tamamen devre dışı bırakıldı) **58 testin hepsini yeşil
bıraktı**. `test_cors_env_var_is_parsed_as_comma_separated_list` yalnızca fonksiyonu
çağırıyor, middleware'e ulaşıp ulaşmadığını doğrulamıyor;
`test_cors_allows_vite_origin_and_is_not_wildcard` ise sadece VARSAYILAN origin'i
kontrol ediyor. Faz 7'nin bitti kriteri ("cross-origin hatasız") tam olarak bu
kablolamaya bağlı ve şu anda regresyona karşı korumasız.
*Ek olarak hayatta kalan:* M23 (`allow_methods=["*"]`) ve M24 (`allow_credentials=True`)
— docstring'de açıkça iddia edilen iki güvenlik kararı da test edilmiyor. K10
iddiasının 4 bileşeninden 3'ü doğrulanmamış durumda.
*Yön (kod değil):* alt süreçte `ALLOWED_ORIGINS` set edilerek kaldırılan gerçek bir
sunucuya preflight atan test, ya da `importlib.reload` ile app'i yeniden kuran test.

**MAJOR-2 — `ALLOWED_ORIGINS="*"` gerçek wildcard üretiyor, engel yok** *(Codex C3, doğrulandı)*
`app/main.py:68-75` + `104-112`. Gerçek sunucuyla teyit edildi: `ALLOWED_ORIGINS=*` →
rastgele `https://evil.example` origin'ine `access-control-allow-origin: *` dönüyor
(hem preflight hem basit istekte). `main.py:25-29`'daki docstring "wildcard kullanılmaz
(K10)" diyor ama bu bir **yorum**, kodda karşılığı yok. Faz 7'de HF Spaces panelinde
"çalışsın diye" `*` yazılması çok olası bir senaryo.

**MAJOR-3 — 34 feature'ın 16'sı ölü; örnek isteğin 10 alanından 4'ü etkisiz (ürün kararı)**
Faz 1 kodunda hata değil, ama Faz 3-4 doğrudan buna çarpacak: `/predict` her zaman
16 adet tam `0.0` SHAP değeri döndürüyor (ölçüldü). `model.py:394-395` "tüm
feature'ları döndür, kaçını göstereceğine frontend karar verir" diyerek bu yükü
Faz 4'e devrediyor — ama karar verilmemiş. Ayrıca demoda gösterilecek formda
`incident_type` / `collision_type` / `number_of_vehicles_involved` /
`police_report_available` alanları değiştirildiğinde sonuç HİÇ değişmeyecek; bu,
"bu adam bana ürün teslim edebilir" demosunda kötü görünür.

### MINOR

**MINOR-1 — `test_shap_positive_class_selection_matches_probability_direction` TOTOLOJİK**
`tests/test_api.py:389-402`. `base_value`'nun girdiden bağımsız sabit olduğu ölçüldü
(5 farklı payload → tek değer). Toplanabilirlik (`sum(shap) = logit(p) - base`)
`_verify_shap_additivity` ile açılışta zaten garanti altında olduğundan,
`total(high) > total(low)` ile `p_high > p_low` **matematiksel olarak eşdeğerdir**.
Yani bu assert toplanabilirlik sağlandığı sürece asla kırmızıya dönemez;
toplanabilirlik bozulduğunda ise uygulama zaten açılmaz (M7/M17'de 48 error).
Test bir şey doğrulamıyor — kaldırılmalı ya da gerçekten bağımsız bir şey ölçmeli.

**MINOR-2 — 422 doğrulama hatası gövdesi gönderilen değeri geri yansıtıyor**
FastAPI varsayılan `RequestValidationError` işleyicisi. Ölçülen:
`POST /predict {"policy_number":521585,"insured_zip":466132}` →
`{"detail":[{"type":"extra_forbidden","loc":["body","policy_number"],"input":521585},...]}`.
İstemci PII gönderirse aynı PII 422 gövdesinde geri döner ve sunucu loglarına düşer.
Veri istemcinin kendisinden geldiği için klasik anlamda sızıntı değil, ama
`test_api.py:410-423` sadece 200 yanıtlarını tarıyor; sözleşmedeki "hassas alan
sızdırılmıyor mu" maddesi hata yolunda doğrulanmamış.

**MINOR-3 — İç mimari açıklamaları public OpenAPI şemasında yayınlanıyor**
`/openapi.json` (ve `/docs`, `/redoc` — üçü de 200) içinde:
- `paths./model-info.get.description` → `main.py:157-164` docstring'i birebir
  yayınlanıyor ve içinde **`preprocessing_contract`, `model_params`,
  `dataset.source_file`** adları geçiyor — yani beyaz listeden bilerek çıkarılan
  alanların adları, dışlanma gerekçeleriyle birlikte public API dokümanında.
- `components.schemas.PredictRequest.description` → `incident_date`, `policy_bind_date`.
- `components.schemas.DatasetInfo.description` → "`source_file` BİLEREK yok".

`schemas.py:369-374` bu alanları "sadece iç mimariyi ifşa eder" diye dışladığını
söylüyor; adları ve gerekçeleri OpenAPI'de yayınlamak bu niyetle çelişiyor.
`test_model_info_does_not_leak_internal_keys` sadece `/model-info` gövdesini tarıyor,
`/openapi.json`'ı taramıyor. Ayrıca müşteriye gösterilecek Swagger UI'da "K2", "K9",
"Elle `dict.pop()` yapmıyoruz..." gibi iç inceleme notlarının görünmesi demo kalitesi
açısından da istenmez.

**MINOR-4 — Ön işleme her istekte İKİ KEZ çalışıyor**
`model.py:386` (`self.pipeline.predict_proba(frame)` → içeride `preprocessor.transform`)
+ `model.py:367` (`self._preprocessor.transform(frame)`). Sayaçla ölçüldü:
`preprocessor.transform` **predict başına 2 kez** çağrılıyor. Ölçülen maliyet
~36.6 ms/istek. Doğruluk sorunu yok, sadece gereksiz iş.

**MINOR-5 — `ALLOWED_ORIGINS=""` sessizce tüm tarayıcı trafiğini kapatıyor**
`main.py:74-75`. Boş parçalar atıldığı için sonuç `[]` oluyor; uygulama açılıyor,
`/health` yeşil, ama hiçbir origin'e izin yok. Açılışta ne hata ne uyarı var. Projenin
geri kalanı fail-fast uyguluyor (artefakt yoksa açılmıyor); burada aynı disiplin
uygulanmıyor. Faz 7'de teşhisi zor bir arıza olur. *(Codex C5 ile örtüşüyor.)*

**MINOR-6 — İstek gövdesi boyut sınırı yok**
20 MB'lık bir gövde kabul edilip tamamen belleğe alındıktan sonra 422 dönüyor
(ölçüldü: HTTP 422, 0.13 s). Kimlik doğrulama ve hız sınırı da olmadığı için public
HF Space'te ucuz bir bellek/CPU tüketim vektörü.

**MINOR-7 — "Sınır eğitim aralığına eşit değil" iddiası tek taraflı çakışmayı kaçırıyor**
`schemas.py:22-25` ve `test_api.py:479-511`. Test `(low, high) != (train_min, train_max)`
karşılaştırması yaptığı için, alt sınırı eğitim minimumuna **eşit** olan 8 alanı
(bkz. (d-1)) "sorunsuz" sayıyor. Bu alanlarda Faz 2 guardrail'i alt yönde asla
tetiklenemez. Çoğu doğal taban olduğu için kabul edilebilir, ama belgelenmemiş.

**MINOR-8 — CLAUDE.md kendi içinde tutarsız: `shap_value` vs `value`**
CLAUDE.md:21 `shap_value`, CLAUDE.md:88 `value`. Implementasyon `value` kullanıyor
(sözleşme bloğuna uygun, doğrusu bu). Faz 4 frontend'i CLAUDE.md:21'i okuyup
`shap_value` beklerse boş grafik çizer. CLAUDE.md düzeltilmeli.

**MINOR-9 — Eş zamanlılık davranışının regresyon testi yok**
M8 (`_shap_lock` kaldırıldı) 58 testi yeşil bıraktı. Ölçümlerim kilidin bu kurulumda
gereksiz olduğunu gösteriyor (1580 eş zamanlı çağrıda 0 sapma), ama
`model.py:228-234`'teki gerekçe kodda bir iddia olarak duruyor ve doğrulanmıyor.

### NIT

**NIT-1 — `/model-info` tam bağımlılık sürümlerini yayınlıyor** *(Codex C4)*
`library_versions` = `{python: 3.14.6, pandas, numpy, scikit-learn, lightgbm, joblib, shap}`.
Model card için savunulabilir (yeniden üretilebilirlik beyanı), ama public bir uçta
çalışan sürümleri duyurmak CVE hedeflemesini kolaylaştırır. Bilinçli karar olmalı.

**NIT-2 — `mypy` yok, `ruff format` zorunlu değil**
Reviewer sözleşmesi `mypy`den bahsediyor ama ortamda kurulu değil ve projede
yapılandırma yok. `ruff format --check` inceleme dosyalarından 3'ünü yeniden
biçimlendirirdi (`app/model.py`, `app/schemas.py`, `tests/test_api.py`).
`ruff check` yeşil olduğu için FAIL değil, ama CI kurulurken netleşmeli.

**NIT-3 — pytest yapılandırma dosyası yok**
`pytest.ini` / `pyproject.toml` / `setup.cfg` yok. Bugün hem repo kökünden hem
`backend/`'den çalışıyor (rootdir + `tests/__init__.py` sayesinde), ama bu pytest'in
çıkarım davranışına bağlı. CI'da açık `pythonpath`/`testpaths` daha sağlam olur.

**NIT-4 — `README.md` bayat**
Rozetler `Python 3.10+` ve **Streamlit** diyor; proje artık FastAPI + React.
Faz 1 kapsamı dışı ama demo repo'sunun ilk görünen yüzü.

---

## Adım 2 — Codex CLI İkinci Görüşü

`codex exec "<inline bağlam>" --sandbox read-only --skip-git-repo-check`
(codex-cli 0.145.0). Dosyaların untracked olduğu, `out_of_distribution_warnings`'in
Faz 1'de kasten boş olduğu, Pydantic sınırlarının kasten geniş olduğu ve
Dockerfile'ın Faz 7 işi olduğu prompt'ta açıkça belirtildi. Rapor formatı (3 başlık,
patch yok, dosya+satır zorunlu) dayatıldı.

### Codex çıktısı (birebir)

> ## 1. Bugs
>
> - **High — declared Python support is incompatible with the shipped artifact.**
>   [backend/requirements.txt:11-15] states that `pipeline.pkl` was created under
>   Python 3.14.6 and requires a 3.14 Docker base, despite the project requirement
>   being Python 3.11+. A Python 3.11 deployment is therefore unsupported and may
>   fail to deserialize the model or behave differently.
>
> - **Medium — the fail-fast startup test does not enter the TestClient lifespan.**
>   [backend/tests/test_api.py:88-98] constructs `TestClient(app)` but never enters
>   it as a context manager. As documented by the project's own fixture, lifespan
>   startup occurs on context entry; therefore the missing metadata is never loaded
>   and the expected `ArtifactError` is not exercised. This test will fail rather
>   than validate fail-fast behavior.
>
> ## 2. Security
>
> - **Medium — a wildcard CORS policy can still be enabled through configuration.**
>   [backend/app/main.py:68-75] accepts every non-empty `ALLOWED_ORIGINS` entry
>   unchanged, including `"*"`, and [backend/app/main.py:104-112] passes it directly
>   to CORS middleware. An accidental `ALLOWED_ORIGINS=*` production setting permits
>   any website to invoke the public scoring API from a visitor's browser, contrary
>   to the stated whitelist intent.
>
> - **Low — `/model-info` exposes exact dependency versions.**
>   [backend/app/main.py:165-189] returns `library_versions`, and
>   [backend/app/schemas.py:510-527] publishes it. Exact runtime-library versions are
>   unnecessary for the stated model-card contract and help attackers target
>   dependency-specific vulnerabilities.
>
> ## 3. UI/UX Observations
>
> - **Medium — the default CORS configuration makes the intended Netlify frontend
>   unusable unless deployment configuration is manually supplied.**
>   [backend/app/main.py:62-75] defaults solely to `http://localhost:5173`. If
>   `ALLOWED_ORIGINS` is absent or misconfigured in Hugging Face Spaces, browser
>   requests from the Netlify production origin will be blocked despite the API
>   otherwise being healthy.

### Reviewer'ın Codex bulgularını doğrulaması

| # | Codex bulgusu | Verdict | Gerekçe |
|---|---|---|---|
| C1 | Python 3.14 vs "3.11+" uyumsuzluğu (High) | **KISMEN** | CLAUDE.md "Python 3.11+" = "en az 3.11"; 3.14.6 bunu **sağlıyor**, çelişki yok. `requirements.txt:11-15` zaten "3.14-slim kullanılmalı, 3.11 garanti değil" diye açıkça uyarıyor — bilinen ve belgelenmiş bir kısıt, bulunmuş bir hata değil. **Ama altında gerçek bir nokta var:** `ModelBundle.load()` çalışan Python/kütüphane sürümlerini `metadata.library_versions` ile **karşılaştırmıyor**; pinlemenin tüm gerekçesi pickle uyumluluğu olduğu hâlde otomatik kontrol yok. (Kısmi savunma: `_verify_shap_additivity` kaba bir davranış sapmasını açılışta yakalar.) Ayrıca `README.md` hâlâ "Python 3.10+" diyor → NIT-4. Severity High DEĞİL, NIT seviyesi. |
| C2 | Fail-fast testi TestClient lifespan'ine girmiyor (Medium) | **YANLIŞ POZİTİF** | Codex satırı yanlış okumuş. `test_api.py:97`: `with pytest.raises(model_module.ArtifactError), TestClient(app):` — bu **iki context manager'lı tek bir `with`**; `TestClient(app)` `__enter__` ediliyor, lifespan çalışıyor. Mantıksal kanıt: girilmeseydi `ArtifactError` hiç fırlamaz ve `pytest.raises` "DID NOT RAISE" ile **kırmızı** olurdu — oysa test geçiyor. Deneysel kanıt: M11b mutasyonu (fail-fast'i kaldır) bu testi **KIRMIZI** yaptı. Test doğru ve öldürücü. |
| C3 | `ALLOWED_ORIGINS=*` wildcard'a izin veriyor (Medium) | **GERÇEK** | Gerçek sunucuyla doğrulandı: `ALLOWED_ORIGINS=*` ile `https://evil.example` origin'ine hem preflight hem basit istekte `access-control-allow-origin: *` dönüyor. Kodun kendi K10 iddiasıyla çelişiyor, hiçbir test/guard engellemiyor. → **MAJOR-2** |
| C4 | `/model-info` tam bağımlılık sürümlerini ifşa ediyor (Low) | **KISMEN** | Olgusal olarak doğru (`python 3.14.6` dahil 7 sürüm dönüyor). Ancak bu bir model card ve sürüm beyanı yeniden üretilebilirliğin **amacı**; "gereksiz" demek fazla iddialı. Gerçek etki düşük (bilgi ifşası, doğrudan istismar değil). → **NIT-1** |
| C5 | Varsayılan CORS Netlify'ı dışarıda bırakıyor (UI/UX, Medium) | **KISMEN** | Bu davranış **kasıtlı ve belgeli** (`main.py:62-64`: Faz 7'de ALLOWED_ORIGINS'e Netlify eklenecek). "Bug" değil. Ama altındaki asıl mesele geçerli: yanlış/eksik yapılandırma **sessizce** başarısız oluyor, açılışta hiçbir sinyal yok. → **MINOR-5** |

**Özet: 5 bulgudan 1'i gerçek (C3), 1'i yanlış pozitif (C2), 3'ü kısmen geçerli.**

### Codex'in KAÇIRDIĞI, reviewer'ın bulduğu maddeler

Codex mutasyon testi yapmadı, sunucu kaldırmadı, eş zamanlı yük denemedi ve booster'ı
incelemedi; bu yüzden aşağıdakilerin hiçbirini göremedi:

1. **MAJOR-1** — CORS env→middleware kablolamasının test edilmediği (M21 hayatta kaldı).
   Codex CORS'a iki kez değindi ama **test kapsamı boşluğunu** görmedi; Faz 7 açısından
   en riskli madde bu.
2. **MAJOR-3** — 16/34 feature'ın ölü olduğu, örnek isteğin 4 alanının hiçbir etkisi
   olmadığı, her yanıtta 16 adet `0.0` SHAP değeri döndüğü.
3. **MINOR-1** — `test_shap_positive_class_selection_matches_probability_direction`'ın
   totolojik olduğu (prompt'ta "tautological assertions" açıkça istendiği hâlde bulamadı).
4. **MINOR-2** — 422 gövdesinin gönderilen PII'yi geri yansıttığı.
5. **MINOR-3** — İç mimari notlarının (`preprocessing_contract`, `model_params`,
   `source_file` adları dahil) public OpenAPI/Swagger'da yayınlandığı — Codex
   `library_versions`'ı bulup bunu kaçırdı, oysa bu daha doğrudan bir iç detay ifşası.
6. **MINOR-4** — Ön işlemenin istek başına iki kez çalıştığı (sayaçla ölçüldü).
7. **MINOR-6** — Gövde boyut sınırı olmadığı (20 MB kabul ediliyor).
8. **MINOR-7 / (d-1)** — 8 alanda alt sınırın eğitim minimumuna eşit olduğu,
   dolayısıyla Faz 2 guardrail'inin o yönde tetiklenemeyeceği.
9. **(d-2)** — Kategorik alanlarda guardrail'in tanım gereği hiç çalışamayacağı.
10. **MINOR-8** — CLAUDE.md'nin kendi içindeki `shap_value` / `value` tutarsızlığı.
11. **M23/M24** — `allow_methods` ve `allow_credentials` iddialarının test edilmediği.

---

## Sonuç

**Faz 1 bitti kriteri: SAĞLANIYOR.**
`/predict`, `/health`, `/model-info` gerçek uvicorn sunucusunda gerçek HTTP ile
çalışıyor; pytest 58/58 yeşil; `ruff check` temiz; API sözleşmesi birebir uyumlu
(fazla/eksik alan yok); PII değeri hiçbir yanıtta yok; elle encoding sözlüğü yok;
eş zamanlı yük altında (1580 çağrı) sonuçlar tutarlı; fail-fast gerçek dosya
kaldırılarak doğrulandı; mutasyon skoru %83.

**BLOCKER yok — Faz 2'ye geçilebilir.**

Faz 2 (guardrails) başlamadan kapatılmalı: **MAJOR-2** (wildcard guard'ı),
**MINOR-7 / (d-1) / (d-2)** (guardrail'in gerçek kapsamının belgelenmesi),
**MINOR-8** (CLAUDE.md `shap_value` → `value`).
Faz 3-4 başlamadan karara bağlanmalı: **MAJOR-3** (ölü feature'lar).
Faz 7 başlamadan kapatılmalı: **MAJOR-1** (CORS kablolama testi), **MINOR-5**, **MINOR-6**.

**Düzeltmeler `backend-agent` tarafından yapılacaktır. Reviewer hiçbir kaynak
dosyayı değiştirmemiştir** — mutasyon testi sırasında geçici olarak değiştirilen tüm
dosyalar yedekten bayt bazında geri yüklenmiş, sha256 eşleşmesi ve `58 passed`
yeniden doğrulanmıştır (bkz. Adım 1 §7).
