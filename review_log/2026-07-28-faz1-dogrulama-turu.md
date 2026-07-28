# Faz 1 — Doğrulama Turu (C bölümü) · Review Log

| | |
|---|---|
| **Tarih** | 2026-07-28 |
| **Branch** | `feat/phase-1-fastapi-api` |
| **Faz** | 1 — FastAPI endpoint'leri + Pydantic validation |
| **Bu turun konusu** | 2026-07-27'de **yarım kalan** doğrulama turunun C bölümü |
| **Önceki kayıtlar** | `2026-07-27-faz1-codex-review.md` (ilk tur, PASS), `2026-07-27-faz0-codex-review.md` |
| **Başlangıç durumu** | 94 test yeşil, çalışma ağacı temiz (`c4eb711`) |
| **Bitiş durumu** | 102 test yeşil, 8 yeni test, 5 mutasyon öldürüldü, 3 gerçek bug düzeltildi |
| **İkinci görüş** | Codex CLI 0.145.0 · `--sandbox read-only` · 6 bulgu |

---

## 0. Neden bu tur gerekliydi

2026-07-27'de Faz 1 `c4eb711` olarak commit'lendi ama doğrulama turu tamamlanmadı
ve raporu **bilerek yazılmadı** — eksik doğrulamayı tam gibi kayda geçirmemek
için. Reviewer'ın o günkü kararı "YETERSİZ VERİ" idi.

Kapanmamış maddeler şunlardı:

1. Determinizm yeniden-eğitimi
2. Fairness metinlerinin aşırı-iddia denetimi
3. `feature_influence` beyaz liste sızıntı testi
4. Eş zamanlılık regresyon testi (mutasyon A-M8 iki turdur hayatta kalıyordu)
5. `_verify_transform_equivalence()` çağrısının silinmesinin yakalanmaması
6. `test_shap_values_match_an_independent_oracle` docstring'inin aşırı iddialı olması
7. Codex CLI ikinci görüşü

Bu tur yedisini de kapatıyor.

---

## 1. Temiz baseline (mutasyon kalıntısı taraması)

Önceki turda kaynak dosyalar **yerinde** mutasyona uğratılmıştı; ilk iş diskteki
kodun bozuk olmadığını kanıtlamaktı.

| Kontrol | Sonuç |
|---|---|
| `git status` | temiz — diskteki kod = `c4eb711` |
| `pytest` × 3 art arda | 94 passed, 94 passed, 94 passed (sapma yok) |
| `ruff check .` | `All checks passed!` (uvx ruff 0.14.5) |
| Kaynak dosya incelemesi | `main.py` / `model.py` / `schemas.py` elle okundu, mutasyon kalıntısı yok |

**Süreç bulgusu — sha256 baseline'ı Windows'ta yanıltıcı.** Mutasyonları geri
alırken `schemas.py`'ın sha256'sı baseline'a dönmedi
(`1943E0…` → `71A881…`). İçerik kaybı **yoktu**: `git hash-object` çıktısı
HEAD'deki blob ile birebir aynıydı (`ef1df05…`). Sebep `core.autocrlf=true` —
`git checkout` dosyayı CRLF ile geri yazdı, içerik aynı kaldı ama bayt
temsili değişti. Bu ortamda mutasyon geri-alma kanıtı **`git hash-object` /
`git status`** ile yapılmalı, sha256 ile değil.

---

## 2. Yeni testler ve mutasyonla ölüm kanıtı

Dört test eklendi. Her biri, iddia ettiği korumayı bozan bir mutasyonla
**fiilen** sınandı; mutasyon uygulanıp test koşuldu ve `git checkout` ile geri
alındı.

### A-M8 — SHAP kilidi (iki turdur hayatta kalıyordu)

`test_shap_lock_is_actually_held_during_explanation`

Neden eski test öldüremiyordu: `test_concurrent_predictions_are_bitwise_identical`
**sonuç** bazlı. shap'in her çağrıda yeniden atadığı `expected_value` hep aynı
değere ayarlandığı için yarış durumu çıktıda gözlemlenebilir fark üretmiyor.
Yani kilit bugünkü bir bug'ı değil, kütüphanenin paylaşılan durumu mutasyona
uğratma davranışını kapatıyor. Sonuç bazlı hiçbir test bunu kanıtlayamaz.

Yeni test korumayı doğrudan gözlemliyor: explainer sarmalanıp çağrı anında
`_shap_lock.locked()` okunuyor.

```
Mutasyon: model.py `with self._shap_lock:` -> `if True:`
Sonuç   : FAILED test_shap_lock_is_actually_held_during_explanation  ✅ ÖLDÜ
Geri alma doğrulandı (git hash-object değişmedi)
```

### Açılış doğrulamalarının atlanması

`test_load_runs_every_startup_verification`

`_verify_transform_equivalence()` **çağrısını** `load()` içinden silmek tüm
suite'i yeşil bırakıyordu: artefakt zaten tutarlı olduğu için doğrulamayı
atlamanın gözlemlenebilir etkisi yok. Ama koruma tam da tutarsız artefakt
içindir.

```
Mutasyon: model.py load() içinden `bundle._verify_transform_equivalence()` silindi
Sonuç   : FAILED test_load_runs_every_startup_verification  ✅ ÖLDÜ
```

### Beyaz listenin geleceğe dönük vaadi

`test_model_info_projection_drops_unknown_nested_keys`

Mevcut sızıntı testleri **bugün var olan** üç anahtarı arıyordu. Beyaz listenin
asıl vaadi ise "yarın metadata'ya eklenen alan, açıkça izin verilene kadar
dışarı çıkmaz". Yeni test metadata'nın üç ayrı iç içe seviyesine sahte alanlar
enjekte edip projeksiyonu koşuyor.

```
Mutasyon: schemas.py FeatureInfluenceEntry `extra="ignore"` -> `extra="allow"`
Sonuç   : FAILED test_model_info_projection_drops_unknown_nested_keys  ✅ ÖLDÜ
```

### Ölü feature iddiasının uçtan uca kanıtı

`test_dead_features_change_neither_the_score_nor_their_own_shap_value`

Bu bir mutasyon kapatma değil, **fairness metnindeki iddianın** kanıtı — bkz.
bölüm 4.

### Hayatta kalan mutasyon (dürüstlük kaydı)

`_verify_transform_equivalence`'ın **gövdesi** boşaltıldığında (başına `return`
konulduğunda) 98 testin tamamı yeşil kaldı. Yeni test **çağrının yapıldığını**
bağlıyor, çağrılan metodun içinin doğru olduğunu değil.

Kısmi savunma var: aynı iddiayı `test_split_transform_matches_full_pipeline_bitwise`
test seviyesinde ayrıca kontrol ediyor. Yani gövde boşalırsa CI yakalar; kaybolan
şey **açılış anındaki fail-fast**'tir. Bu, Codex C-3'te ortaya çıkan yapısal
sorunun aynısı: "doğrulama çağrıldı mı?" ile "doğrulama işe yarıyor mu?" farklı
sorular. C-3 için doğrulamanın kendisi güçlendirildi (bölüm 9); burada durum
kayda geçirildi ve kapatılmadı — kabul edilen kalıntı risk.

### Süreç kaydı: `git checkout` düzeltmeleri de siliyor

Mutasyon geri alınırken `git checkout -- app/main.py app/model.py` çalıştırıldı
ve **commit'lenmemiş düzeltmeler de silindi**. Düzeltmeler yeniden uygulandı
(mutasyon sonucu zaten alınmıştı). Ders: mutasyon turu ya temiz ağaçta
koşulmalı ya da geri alma `git checkout` yerine hedefli `Edit` ile yapılmalı.

---

## 3. Determinizm

`train_pipeline.py --output-dir ../.determinism_check` ile mevcut artefaktlara
**dokunmadan** yeniden eğitildi ve karşılaştırıldı.

| Karşılaştırma | Sonuç |
|---|---|
| `pipeline.pkl` sha256 | `CA80B84F…` = `CA80B84F…` — **bit bazında aynı** |
| `metadata.json` (`trained_at` hariç) | **birebir aynı** |
| `test_pr_auc` | `0.6527088560280899` (değişmedi) |
| Ölü feature listesi | aynı 16 feature |

HPO yeniden koşulmuyor (Faz 0 kararı): `LGBM_PARAMS` sabit, `random_state=42`.
Bu yüzden CLAUDE.md checklist'indeki "Optuna `TPESampler(seed=…)`" maddesi bu
fazda **uygulanamaz** — Optuna devrede değil, determinizm doğrudan bayt
karşılaştırmasıyla kanıtlandı.

Geçici dizin koşu sonunda silindi.

---

## 4. Fairness metinlerinin aşırı-iddia denetimi

`metadata.json -> fairness` bölümü baştan sona okundu. Bu metin `/model-info`
üzerinden sigorta müşterisine gösterilecek.

**Aklayıcı iddia bulunmadı.** `status: "declared_not_audited"`,
`audit_performed: false`, `audit_metrics_computed: []` ve dört maddelik
`production_requirements` yerinde. `notes` ve `field_semantics` ölçümün
sınırlarını açıkça yazıyor: yeniden eğitimde değişebilir, vekil feature'lar
sinyali geri taşıyabilir, grup bazlı metrik hesaplanmadı.

**Denetlenen tek somut teknik iddia:**

> "split_count=0 olan bir feature bu eğitilmiş modelde hiçbir tahmini etkilemez
> ve SHAP katkısı her zaman tam 0.0'dır."

Bu iddia **doğru ama tek örnekle test edilmişti**. Mevcut
`test_dead_features_have_exactly_zero_shap_value` yalnızca CLAUDE.md örnek
isteği üzerinde ölçüyordu; "her zaman" tek girdiyle kanıtlanmaz.

Yeni test her ölü feature'ın eğitimde görülen **değer uçlarını** tek tek
deniyor (>40 kombinasyon) ve iki şeyi birden bağlıyor: skor bit bazında sabit
kalıyor **ve** o feature'ın kendi SHAP katkısı tam 0.0. Test ayrıca ters yönü
de kontrol ediyor — canlı bir feature (`insured_hobbies`) skoru değiştirmeli,
yoksa eşitlikler "model hiçbir şeye tepki vermiyor"dan ibaret olurdu.

Gerçek HTTP üzerinden de doğrulandı:

```
insured_sex=FEMALE -> 0.7299506078527795
insured_sex=MALE   -> 0.7299506078527795   (eşit)
```

Yani "cinsiyeti değiştirmek skoru değiştirmiyor" artık beyan değil ölçüm.
**Bu bir fairness denetimi değildir** ve metin de böyle iddia etmiyor.

---

## 5. Aşırı iddialı docstring düzeltildi

`test_shap_values_match_an_independent_oracle` →
`test_shap_values_stay_aligned_with_the_booster_contributions`

Eski docstring oracle'ın **bağımsız** olduğunu iddia ediyordu. Doğrulandı ki
değil: `shap/explainers/_tree.py:625` LightGBM dalında hesabı kendisi yapmıyor,
`original_model.predict(X, pred_contrib=True)` çağırarak doğrudan booster'a
delege ediyor. Karşılaştırılan iki taraf aynı sayısal kaynaktan besleniyor.

Test **değerli ve korunuyor** — ama artık doğru şeyi iddia ediyor: SHAP'in
matematiğini değil, kendi eşleme katmanımızı (`zip`, `abs`'e göre sıralama,
pozitif sınıf ekseni) bağlıyor. Bunlar katkıları adlardan koparabilir ve sonuç
yine "geçerli" görünür.

---

## 6. Gerçek sunucu doğrulaması (TestClient değil)

`python -m uvicorn app.main:app --port 8742` ile ayağa kaldırıldı, istekler
gerçek HTTP ile atıldı.

| Kontrol | Sonuç |
|---|---|
| `GET /health` | 200 · `{"status":"ok","model_loaded":true,"model_version":"1.0.0"}` |
| `POST /predict` | 200 · `prob=0.7299506078527795` `risk=high` `shap_n=34` `ood=0` |
| Yanıt alanları | tam olarak sözleşmedeki 4 alan, fazlası yok |
| `GET /model-info` | 200 · 17.934 bayt |
| PII kapsamı | `policy_number` / `insured_zip` / `incident_location` **yalnızca** `dropped_columns` beyanında |
| `/openapi.json` | `preprocessing_contract`, `model_params`, `source_file`, `n_estimators` — **hiçbiri yok** |
| 422 gövdesi | girdi değeri yansıtılmıyor (`loc`/`msg`/`type` korunuyor) |
| 413 gövde sınırı | 70 KB gövde → 413 |
| **Eş zamanlılık** | **200 paralel gerçek HTTP isteği → 1 farklı olasılık, 1 farklı SHAP gövdesi** |

Eş zamanlılık sondası `Start-ThreadJob` bu PowerShell 5.1 kurulumunda
bulunmadığı için Python `ThreadPoolExecutor` + `urllib` ile koşuldu.

---

## 7. CLAUDE.md "Bilinen Hata Checklist"

- [x] Optuna `TPESampler(seed=…)` — **uygulanamaz**, bu fazda HPO koşulmuyor; determinizm bayt bazında kanıtlandı (bölüm 3)
- [x] Train/test split feature engineering'den ÖNCE — `train_pipeline.py:777-781`, imputer/encoder yalnızca `X_train`'den öğreniyor
- [x] Elle encoding sözlüğüne dönülmedi — `model.py` yalnızca şema normalizasyonu yapıyor, encoding baştan sona `Pipeline` içinde
- [x] SHAP versiyonu pinli — `requirements.txt`: `shap==0.52.0` (tüm bağımlılıklar `==` ile pinli)
- [ ] Guardrail "unseen" değerde tetikleniyor mu — **Faz 2'nin konusu**, Faz 1'de `out_of_distribution_warnings` her zaman boş (sahte uyarı üretilmiyor)
- [x] `/predict` yanıtında hassas/gereksiz alan yok — bölüm 6

---

## 8. `ruff format` kararı

İki turdur "5 dosya yeniden biçimlenirdi" notu duruyordu. Bu tur **karara
bağlandı: uygulanmayacak.**

`ruff check` yeşil ve projede format zorunluluğu yok. `ruff format` çıktısı
incelendiğinde okunabilirliği **düşürdüğü** görüldü — `line-length 100` ile bile
215 satırlık diff üretiyor ve şuna benzer sonuçlar veriyor:

```python
# mevcut (elle yazılmış, okunur)
collision_type: (
    Literal["Front Collision", "Rear Collision", "Side Collision", "?"] | None
) = None

# ruff format çıktısı
collision_type: Literal["Front Collision", "Rear Collision", "Side Collision", "?"] | None = (
    None
)
```

Not artık "yapılmadı" değil, "bilinçli olarak uygulanmıyor" olarak kapatıldı.
İleride bir CI format gate'i istenirse `ruff.toml`'a `line-length` eklenip
tek seferde uygulanabilir.

---

## 9. Codex CLI ikinci görüşü

`codex exec --sandbox read-only`, inline brief ile (repoda `AGENTS.md` yok —
CLAUDE.md'deki konvansiyon). Altı bulgu verdi. **Hiçbiri olduğu gibi kabul
edilmedi**; ikisi bağımsız bir sonda scriptiyle reproduce edildi.

### C-1 [ÖNEMLİ · DOĞRULANDI · DÜZELTİLDİ] — Gövde sınırı, gövdeyi okumayan endpoint'lerde aşılabiliyordu

Sınırın akış bazlı kolu downstream'in `receive()` çağırmasına bağlıydı.
`/health` ve `/model-info` istek gövdesini hiç okumaz.

```
DÜZELTME ÖNCESİ:  GET /health  (160 KB chunked) -> 200   ✗
                  GET /health  (100 KB, Content-Length VAR) -> 413  ✓
```

Ucuz yol (`Content-Length`) zaten çalışıyordu; açık yalnızca başlığı hiç
göndermeyen chunked istemcideydi.

**Düzeltme:** `BODYLESS_METHODS` (GET/HEAD/OPTIONS/DELETE) için middleware
gövdeyi kendisi tüketip sayıyor (`_call_with_drained_body`). Sınır artık
endpoint'in gövdeyle ilgilenip ilgilenmemesinden bağımsız. Normal GET'te maliyet
sıfır — döngü ilk mesajda biter.

```
DÜZELTME SONRASI (gerçek uvicorn + gerçek chunked transfer-encoding):
  GET  /health     (160 KB chunked) -> 413
  GET  /model-info (160 KB chunked) -> 413
  POST /predict    (160 KB chunked) -> 413
  GET  /health     (16 KB chunked)  -> 200
  GET  /health     (gövdesiz)       -> 200
```

Test: `test_body_limit_applies_to_endpoints_that_never_read_the_body`
Mutasyon (`BODYLESS_METHODS` yolu devre dışı) → test kırmızı. ✅

> Kalan sınır: gövdeyi okumayan bir **POST** endpoint'i ileride eklenirse aynı
> boşluk döner. Bugün öyle bir endpoint yok (`/predict` gövdeyi okuyor). Tam
> koruma Faz 7'de ters proxy / ASGI sunucu seviyesinde de kurulmalı.

### C-2 [ÖNEMLİ · DÜZELTİLDİ] — Pozitif sınıf bulunamazsa sessizce tahmin ediliyordu

`_positive_class_index()` `classes_` içinde `1` yoksa "son sınıf"a düşüyordu.
Bu fallback tehlikeliydi: yanlış indeks hem `predict_proba` sütununu hem SHAP
eksenini **aynı anda** kaydırır, dolayısıyla `_verify_shap_additivity` de aynı
yanlış ekseni kullanıp eşitliği sağlar. Yani hata kendi doğrulamasını atlatarak
"ters işaretli ama tutarlı" bir servis açardı.

**Düzeltme:** fallback kaldırıldı, `ArtifactError` fırlatılıyor (`classes_` yoksa
da). Bu projede `classes_` her zaman `[0, 1]` — buraya düşmek artefaktın
beklenmedik şekilde değiştiği anlamına gelir ve doğru davranış açılmamaktır.

Test: `test_missing_positive_class_fails_fast_instead_of_guessing`

### C-3 [ÖNEMLİ · DOĞRULANDI · DÜZELTİLDİ] — En ciddi bulgu: SHAP ad eşlemesi fail-fast korumasızdı

`_verify_feature_alignment` yalnızca **uzunluğa** ve `cat__` / `remainder__`
**önekine** bakıyordu. İkisi de adların SIRASI hakkında hiçbir şey söylemez.

Reproduce edildi — `transformed_display_names` listesinin yalnızca iki elemanı
yer değiştirildi:

```
SONUÇ: açılış doğrulamalarının HEPSİ GEÇTİ -> bozuk metadata ile servis açılır
       skorlar doğru kalıyor, SHAP katkıları YANLIŞ feature adlarıyla yayınlanıyor
```

`_verify_shap_additivity` bunu yakalayamaz: toplam değişmediği için eşitlik
korunur. Testler yakalıyordu
(`test_shap_feature_names_follow_the_transformed_column_order`), **açılış
yakalamıyordu** — yani bozuk bir artefaktla production'a çıkmak mümkündü.

Bu, demonun tam da satmaya çalıştığı şeyin (açıklanabilirlik) sessizce yalan
söylemesi olurdu. Bulguların en ciddisi.

**Düzeltme:** adlar artık metadata'ya güvenilerek değil, pipeline'ın kendi
çıktısından türetilip **pozisyon bazında** karşılaştırılıyor; sapma varsa hata
mesajı hangi pozisyonların uyuşmadığını da yazıyor.

```
DÜZELTME SONRASI: ArtifactError: transformed_display_names, pipeline'ın kolon
sırasıyla eşleşmiyor (2 pozisyon). SHAP katkıları yanlış feature adlarıyla
yayınlanırdı. İlk farklar: ["pozisyon 0: metadata='policy_csl' pipeline='policy_state'", …]
```

Test: `test_display_names_out_of_order_fails_fast`
Mutasyon (kontrol devre dışı) → test kırmızı. ✅

### C-4 [ÖNEMLİ · KABUL EDİLDİ · FAZ 2'YE DEVREDİLDİ] — Kategorik OOD uyarısı API sözleşmesinden erişilemez

Kategorik alanlar `Literal` ile eğitimde görülen değerlerle sınırlı; bilinmeyen
bir kategori guardrail'e **ulaşmadan** 422 alır. Buna karşın
`out_of_distribution_warnings` alanı "eğitim dağılımı dışında kalan alanlar"
diye genel tanımlanmış.

Sonuç: **Faz 2 yalnızca sayısal OOD uyarabilir.** Kategorik OOD için uyarı
üretildiğini varsayan bir UI vaadi sessizce yanlış olur.

Bu bir bug değil, tasarımın sonucu — ve doğru tasarım: bilinmeyen kategori
`OrdinalEncoder`'ın `-1` yoluna düşerdi, 422 dönmek daha güvenli. Ama
**belgelenmemişti**. `schemas.py` sayısal alanlar için "tek yönlü körlük" notunu
zaten içeriyordu; kategorik durum Faz 2 notu olarak kayda geçti (bölüm 11).

### C-5 [MINOR · KARARA BAĞLANDI — değişiklik yok] — `/model-info` çalışma ortamı sürümlerini yayınlıyor

`library_versions` alanı Python, LightGBM, scikit-learn, SHAP sürümlerini
public olarak veriyor. PII veya hiperparametre sızıntısı **değil**; güvenlik
açısından sunucu teknoloji parmak izi sağlıyor.

Bu teknik bir bug değil, ürün/güvenlik dengesi kararı olduğu için düzeltilmedi
ve Nebi'ye soruldu.

**Karar (2026-07-28, Nebi): olduğu gibi kalacak.** Gerekçe: sürümler model
card şeffaflığının parçası ("bu model şu sürümlerle eğitildi, reprodüksiyon
için bunlar gerekli") ve Faz 5 model card sayfasını besliyor. API zaten
kimlik doğrulaması olmayan public bir demo; hassas veri tutmuyor, dolayısıyla
parmak izinin operasyonel karşılığı düşük.

### C-6 [MINOR · DÜZELTİLDİ] — Beyaz liste testi iki alt dalı kapsamıyordu

Yeni projeksiyon testi `feature_influence`, `fairness`, `dataset`, `metrics`
dallarına canary enjekte ediyordu ama `training_ranges` ve `defaults`
atlanmıştı. Kod bugün güvenli (`extra="ignore"`), ama kapsanmayan dal iddiayı
test edilmemiş bırakır.

**Düzeltme:** `test_model_info_projection_covers_training_ranges_and_defaults`.

### Codex'in doğruladıkları (bulgu değil)

- `/predict` istemci girdisini echo etmiyor; 422 handler `input`/`ctx` düşürüyor
- CORS wildcard ve credentials yolları kapalı, ayrı bypass bulunmadı
- Fairness metni aşırı aklayıcı değil — `audit_performed=false`, vekil riskleri,
  yeniden eğitim riski ve grup metriklerinin eksikliği açıkça yazılmış
- "split_count=0 → tahmin etkisi ve TreeSHAP katkısı sıfır" iddiası teknik
  olarak savunulabilir; yeni test örnekleme yapıyor (evrensel matematiksel kanıt
  değil) ama booster split-count doğrulamasıyla birlikte yanıltıcı değil
- Yeni testlerden lock ve startup-call testleri totolojik değil, korudukları
  mekanizmayı doğrudan gözlüyor

---

## 10. Karar

**Faz 1 KAPANDI.**

Gerekçe: 2026-07-27'de açık bırakılan yedi maddenin tamamı kapatıldı, üç gerçek
bug bulunup düzeltildi ve her düzeltme mutasyonla sınandı. `/predict`, `/health`,
`/model-info` gerçek uvicorn üzerinde çalışıyor; 102 test yeşil; `ruff check`
temiz; artefakt üretimi bit bazında deterministik.

Önceki turdan farkı: o tur "test yeşil" diyordu, bu tur **yeşilin ne kanıtladığını**
gösteriyor — ve iki yerde yeşilin hiçbir şey kanıtlamadığı ortaya çıktı (C-1, C-3).

| | |
|---|---|
| Test | 102 passed |
| Lint | `ruff check` temiz |
| Mutasyon | 5 mutasyon uygulandı, 5'i öldürüldü; 1 bilinen kalıntı (bölüm 2) |
| Bug | 3 bulundu, 3 düzeltildi |
| Determinizm | `pipeline.pkl` sha256 birebir |
| Gerçek HTTP | `/health`, `/predict`, `/model-info`, 413, 422, 200× eş zamanlılık |

---

## 11. Açık maddeler ve Faz 2'ye devredilenler

**Karara bağlananlar:**

1. **C-5** — `/model-info` `library_versions` **yayınlamaya devam edecek**
   (Nebi, 2026-07-28). Model card şeffaflığı, Faz 5 sayfasını besliyor.

**Faz 2 (guardrail) tasarımına girdi:**

2. **C-4** — Guardrail **yalnızca sayısal** OOD uyarabilir. Kategorik alanlar
   `Literal` ile kapalı olduğu için bilinmeyen kategori 422 alır, uyarı üretmez.
   `out_of_distribution_warnings` alanının açıklaması ve Faz 4'teki UI banner'ı
   bu sınırı yansıtmalı — "kategorik OOD uyarısı gelir" varsayımı yanlış olur.
3. **Tek yönlü körlük** — sekiz alanda Pydantic alt sınırı eğitim minimumuna
   eşit; o alanlarda "eğitim aralığının altında" yönünde OOD hiç tetiklenemez
   (`schemas.py` girişindeki not). Bu eksiklik değil, fiziksel gerçeğin sonucu.
4. **OOD uyarılarını `has_influence` ile önceliklendir** — modelin hiç
   kullanmadığı 16 ölü alanda uyarı vermek yanıltıcı olur (bkz. Faz 3 "işaretle
   ve göster" kararı).

**Kabul edilen kalıntı risk:**

5. `_verify_transform_equivalence` gövdesinin boşaltılması açılışta yakalanmıyor
   (bölüm 2). Test seviyesinde kapsanıyor, fail-fast seviyesinde kapsanmıyor.
6. Gövdeyi okumayan bir POST endpoint'i ileride eklenirse C-1 boşluğu döner.
   Faz 7'de ters proxy seviyesinde ikinci bir sınır kurulmalı.
