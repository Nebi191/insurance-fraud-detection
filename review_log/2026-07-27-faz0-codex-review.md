# Faz 0 — Pipeline Konsolidasyonu · Review Log

| | |
|---|---|
| **Tarih** | 2026-07-27 |
| **Faz** | 0 — Pipeline konsolidasyonu (preprocessor + model tek pkl) |
| **Bitti Kriteri (CLAUDE.md)** | `pipeline.pkl` tek dosya, eski 2 pkl'ye bağımlılık yok |
| **Reviewer** | `reviewer` agent (Adım 1 iç kontrol) |
| **İkinci görüş** | Codex CLI 0.145.0 · `gpt-5.6-terra` · `--sandbox read-only` |
| **İncelenen dosyalar** | `backend/train_pipeline.py`, `backend/models/metadata.json`, `backend/models/pipeline.pkl`, `backend/requirements.txt`, `.gitignore` |

---

## Adım 1 — İç Kontrol (reviewer, bağımsız koşuldu)

| # | Kontrol | Sonuç |
|---|---|---|
| 1 | Taze process, sadece `pipeline.pkl` + ham DataFrame → `predict_proba` | **PASS** — `Pipeline(['preprocessor','model'])`, uyarısız yüklendi, `fraud_probability=0.729951` (CLAUDE.md örnek payload'ı) |
| 2 | Eski artefakt bağımlılığı (`preprocessor.pkl` / `best_model_lgb.pkl` / `defaults.pkl`) | **PASS (disk)** — üçü de diskte yok; `backend/models/` = `{pipeline.pkl, metadata.json}`. **UYARI:** git'te TRACKED olan kök `app.py` hâlâ bu 3 dosyayı `joblib.load` ediyor (bkz. R-01) |
| 3 | `ruff check backend/` | **ÇALIŞTIRILDI** (ortamda kurulu değil → `uvx ruff@latest 0.16.0`). 4 bulgu, hepsi `RUF046` (kozmetik, gereksiz `int()` cast): `train_pipeline.py:247,379,380,381` |
| 4 | `pytest` | **YOK** — `backend/tests/` boş, hiçbir test dosyası yok, pytest kurulu değil. Faz 0 için beklenen (testler Faz 1'de) |
| 5 | `python backend/train_pipeline.py` uçtan uca | **PASS** — hatasız koştu. Train PR-AUC `0.7890`, Test PR-AUC `0.6527` → `metadata.json` ile birebir aynı |
| 6 | Determinizm | **PASS (güçlü)** — reviewer'ın yeniden ürettiği `pipeline.pkl` committed dosyayla **byte-byte eşit**; 200 rastgele satırda `max\|Δproba\| = 0.0`; metadata'da tek fark `trained_at` |
| 7 | PII (`policy_number`, `insured_zip`, `incident_location`) | **PASS** — sadece `feature_list.dropped_columns` altında *isim* olarak var; `defaults`/`training_ranges`/`raw_input_order`/SHAP adlarında yok |
| 8 | `requirements.txt` pinleri ↔ `pip freeze` | **PASS** — 30/30 pin birebir eşleşti, uydurma pin yok |
| 9 | `.gitignore` negasyonu | **PASS** — `git add --dry-run` `backend/models/pipeline.pkl`'i ekliyor; `*.pkl` kuralı doğru şekilde delinmiş |

**Adım 1 sonucu: PASS** → Adım 2'ye geçildi.

### Reviewer'ın kendi ek bulguları (Codex'in görmediği)

| ID | Dosya | Bulgu |
|---|---|---|
| R-01 | `app.py:9-11` | Git'te **tracked** olan kök Streamlit uygulaması hâlâ `best_model_lgb.pkl` / `preprocessor.pkl` / `defaults.pkl` yüklüyor — üçü de artık yok. Repo'daki tek çalıştırılabilir giriş noktası **kırık**. Faz 0 bitti kriterinin ("eski 2 pkl'ye bağımlılık yok") ruhuna aykırı. |
| R-02 | `.gitignore:7` | `data/` ignore'da → `data/insurance_claims.csv` repo'ya girmiyor. `train_pipeline.py` **temiz bir clone'da koşamaz**; Faz 0'ın yeniden üretilebilirliği yalnızca Nebi'nin makinesinde geçerli. |
| R-03 | `metadata.json` `training_ranges.incident_year` | `incident_year` eğitim verisinde **sabit** (min=max=2015, nunique=1). Sıfır varyanslı feature; Faz 2 guardrail'i 2015 dışındaki **her** değerde OOD uyarısı basacak (demo 2026'da gösteriliyor). |
| R-04 | `metadata.json` `training_ranges.umbrella_limit` | `min = -1000000` (kaynak CSV'de 1 adet kirli satır). Guardrail negatif umbrella limitini "dağılım içi" sayacak. |
| R-05 | pipeline davranışı | Fazladan kolon (ör. `policy_number`) verilince pipeline **sessizce kabul edip yok sayıyor** (proba değişmiyor). Katı şema reddi yok → Faz 1'de Pydantic `extra="forbid"` ile kapatılmalı. |
| R-06 | pipeline davranışı | Sayısal alanda `NaN` **kabul ediliyor** (LightGBM native missing). Sayısal kolonlar için imputer yok — bilinçli mi, kaza mı belirsiz. |
| R-07 | `README.md` | Hâlâ Streamlit uygulamasını ve eski repo yapısını anlatıyor; FastAPI/monorepo yapısıyla uyumsuz. |
| R-08 | kök `requirements.txt` | UTF-16 BOM'lu (mojibake), `backend/requirements.txt` ile çelişiyor. İki ayrı requirements dosyası hangi ortam için, belirsiz. |
| R-09 | repo geneli | `pyproject.toml` / `ruff.toml` yok; `ruff` ve `pytest` hiçbir requirements dosyasında değil. Lint/test ortamı repo'da sabitlenmemiş. |

---

## Adım 2 — Codex CLI İkinci Görüş (ham çıktı, düzenlenmedi)

## Bugs

- `backend/train_pipeline.py:350-353` — Tek sınıflı ya da azınlık sınıfında yalnızca bir örnek bulunan veri setlerinde `stratify=y` split'i başarısız olur; ayrıca pozitif sınıf yoksa `scale_pos_weight` hesaplaması sıfıra bölünür.

- `backend/train_pipeline.py:112-123` — Zorunlu tarih ve `QUESTION_MARK_COLS` alanları doğrudan indeksleniyor. Veri şemasında bu alanlardan biri yoksa kontrollü doğrulama yerine `KeyError` ile işlem kesilir.

- `backend/train_pipeline.py:109, 112-113` — `pd.to_datetime` için hata politikası belirtilmemiş. Bozuk/tanınmayan tarih değerleri eğitim akışını istisna ile durdurabilir; parse edilebilen fakat belirsiz tarih biçimleri de ortama bağlı yorumlanabilir.

- `backend/train_pipeline.py:92-94, 193` — `random_state` sabit olsa da LightGBM'in iş parçacığı/deterministik çalışma ayarları sabitlenmemiş. Farklı makine, CPU ya da paralellik koşullarında aynı eğitim artefaktı ve metrikler garanti edilmez.

- `backend/train_pipeline.py:245-249, 284-289` — Tamamen eksik bir sayısal eğitim kolonu için medyan/min/max `NaN` olur. Python'ın JSON serileştirmesi bunu standart dışı `NaN` değeri olarak yazabilir; metadata'yı katı JSON ayrıştırıcıları okuyamaz.

## Security

- `backend/train_pipeline.py:329-338, 422-425` — Komut satırından verilen `--output-dir` herhangi bir mutlak yola çözümlenip oluşturuluyor ve içine artefakt yazılıyor. Çağıran bağlam güvenilmiyorsa bu, yetkili süreç erişimi kapsamındaki dosyaların üzerine yazılmasına yol açabilir.

- `backend/train_pipeline.py:329, 333, 341` — `--data` ile yerel dosya sistemi üzerinde keyfî bir CSV yolu okunabiliyor. Bu arayüz daha sonra servis/otomasyon bağlamına taşınırsa dosya-okuma yetkisi sınırlandırılmamış olur.

- `backend/train_pipeline.py:424`; `.gitignore:6, 14`; `backend/requirements.txt:4-13` — Commit edilen `pipeline.pkl`, joblib/pickle tabanlıdır. Böyle bir artefaktın yüklenmesi kod çalıştırabilir; incelenen dosyalarda artefakt bütünlüğünü ya da güvenilir kaynağını doğrulayan bir mekanizma görünmüyor.

- `backend/models/metadata.json:216-238, 250-278, 290-308` — Metadata, eğitim verisinden türetilmiş varsayılanları içinde `insured_sex`, eğitim, meslek, hobi ve ilişki gibi hassas/kişisel profil alanlarıyla birlikte yayımlıyor. Doğrudan kimlik tanımlayıcılar düşürülmüş olsa da bu alanlar artefakta açıkça taşınıyor.

## Observations

- `backend/train_pipeline.py:102-123, 345-358, 429-430` — Dosya, preprocessing'in tek Pipeline'da konsolide edildiğini ve `pipeline.pkl`in tek başına yeterli olduğunu söylüyor; ancak tarih-yıl türetme, kolon düşürme, `?` normalizasyonu ve hedef dönüşümü Pipeline dışında `load_raw_frame` içinde kalıyor. Pipeline'ın beklediği girdi, gerçek ham CSV şeması değil bu dış işlemden geçmiş `X`.

- `backend/train_pipeline.py:393-395`; `backend/models/metadata.json:22-56` — Metadata, `raw_input_order` alanını pipeline'ın beklediği sıra olarak tanımlıyor; listede `incident_year` ve `policy_bind_year` var, ham `incident_date` ve `policy_bind_date` yok. "Raw input" adı ile içerik birbirini tutmuyor.

- `backend/train_pipeline.py:105-107` — FE kodunun notebook'ta hiç çalışmadığı ve modelin FE'siz yeniden üretildiği belirtiliyor; buna karşılık tarih alanlarından yıl türetme burada özellik mühendisliği olarak uygulanıyor. "FE yok" ifadesinin kapsamı belirsiz.

- `backend/train_pipeline.py:378`; `backend/models/metadata.json:7` — Metadata yalnızca kaynak dosyanın adını kaydediyor. Aynı adlı fakat farklı içerikte bir veri setiyle eğitimin izlenebilirliği sağlanamıyor.

- `backend/requirements.txt:31-35` — FastAPI, Pydantic, Uvicorn ve Starlette hem doğrudan hem bağımlı sürümler olarak pinlenmiş görünüyor; dosya, bu sürüm kombinasyonunun uyumluluğunu belirtmiyor.

---

## Reviewer Değerlendirmesi (Codex bulgularının adjudikasyonu)

| Codex bulgusu | Karar | Gerekçe |
|---|---|---|
| B1 `stratify` / `scale_pos_weight` ÷0 | **Geçerli, düşük** | Sabit 1000 satır / %24.7 pozitif veri setinde ulaşılamaz. Savunmacı sertleştirme, Faz 0 blocker'ı değil. |
| B2 `QUESTION_MARK_COLS` + tarih kolonları korumasız `KeyError` | **Geçerli, düşük** | Gerçek bulgu: `DROP_COLS` `present`/`absent` ile korunuyor (`:115-119`) ama aynı dosyada `QUESTION_MARK_COLS` ve tarih kolonları korunmuyor. **İç tutarsızlık.** |
| B3 `pd.to_datetime` `format=`/`errors=` yok | **Kısmen geçerli, düşük** | `format=` iyi pratik. "Ortama bağlı yorumlanır" iddiası abartılı: 3 koşuda byte-eşit çıktı aldım. |
| B4 LightGBM determinizmi (`deterministic`, `force_row_wise`, `n_jobs`) | **Geçerli ama etkisi sınırlı** | Bu makinede byte-byte determinizmi kanıtladım. Cross-machine riski teorik ve **deploy'u etkilemiyor** — Docker imajı modeli eğitmiyor, repodan kopyalıyor. Yine de "reproducible" iddiası için pinlenmeli. |
| B5 Boş sayısal kolonda `NaN` → geçersiz JSON | **Geçerli, düşük ama gerçek asimetri** | Kod kategorik mod boşsa `ValueError` fırlatıyor (`:240`) ama sayısal medyan `NaN` olursa **hiçbir kontrol yok**. `json.dumps` varsayılan olarak çıplak `NaN` yazar; JS `JSON.parse` bunu reddeder → frontend `/model-info` kırılır. Mevcut veriyle tetiklenmiyor. |
| S1 `--output-dir` keyfî yazma | **False positive** | Geliştiricinin elle çalıştırdığı CLI; argparse yolu saldırı yüzeyi değil, onu çağıran shell ile aynı güven seviyesinde. Aksiyon gerektirmez. |
| S2 `--data` keyfî okuma | **False positive** | Aynı gerekçe. "İleride servise taşınırsa" spekülatif — taşınmıyor. |
| S3 Pickle deserialization, bütünlük doğrulaması yok | **Geçerli — Faz 1/7'ye taşınmalı** | Bugünkü risk düşük (artefakt repo kontrollü). Ama Faz 1'de `model.py` pipeline'ı **yalnızca sabit bir sabit yoldan** yüklemeli; request/env kaynaklı yol asla kabul edilmemeli. |
| S4 `metadata.json` hassas profil alanları yayımlıyor | **Kısmen geçerli — yanlış çerçeveleniyor** | Bunlar **PII kaydı değil, model feature'ı**; `defaults` 800 satırın modu (`insured_sex: "FEMALE"`, `insured_hobbies: "reading"`) — hiçbir birey tanımlanabilir değil. "Sızıntı" nitelemesi yanlış. **Ama altında gerçek bir konu var:** `insured_sex` korunan bir özellik ve model feature'ı; `/model-info` bunu sigorta/fintech müşterisine yayımlayacak. Bu bir *fairness/uyum* sorusu, bug değil → Nebi'nin ürün kararı. Ayrıca Faz 1'de `/model-info`'nun `metadata.json`'ın **tamamını** değil, seçilmiş bir alt kümesini döndürmesi gerekir. |
| O1 Preprocessing'in bir kısmı Pipeline DIŞINDA (`load_raw_frame`) | **Geçerli — raporun en önemli maddesi** | Bağımsız olarak doğruladım: pipeline `feature_names_in_` = `load_raw_frame` sonrası **34 kolon**, ham 39 kolonluk CSV şeması değil. Docstring `:13-15` "`pipeline.predict_proba(raw_dataframe)` dışında bir yol kalmaz" diyor — bu ham CSV satırı için **doğru değil**. *Bitti kriterini ihlal etmiyor* (kriter "tek dosya + eski pkl bağımlılığı yok") ve *"elle encoding yasak" kuralını da ihlal etmiyor* (encoding pipeline içinde). Faz 1'i de bloke etmiyor, çünkü API sözleşmesindeki payload zaten tarih/PII içermiyor. **Ama somut bir Faz 1/2 gereği doğuruyor:** `police_report_available` / `property_damage` / `collision_type` için "?" → NaN dönüşümünü API katmanı kendisi yapmak zorunda; pipeline yapmıyor. |
| O2 `raw_input_order` adı ↔ içeriği uyumsuz | **Geçerli, düşük** | O1'in aynı kökten türevi. İsimlendirme/doküman düzeltmesi. |
| O3 "FE yok" ifadesi ↔ yıl türetme çelişkisi | **Geçerli, düşük** | Doğru. Reviewer'ın R-03'ü bunun daha güçlü hâli: türetilen `incident_year` **sabit** (2015), yani modele sıfır bilgi katıyor. |
| O4 Sadece dosya adı kaydediliyor, hash yok | **Geçerli, düşük** | R-02 ile birleşince daha ağır: `data/` gitignore'da olduğu için veri seti repoda hiç yok — izlenebilirlik değil, **yeniden üretilebilirlik** kayıp. |
| O5 FastAPI/Pydantic/Starlette pin uyumluluğu belirtilmemiş | **False positive** | 30/30 pin `pip freeze` ile birebir eşleşiyor — çözülmüş, tutarlı bir set. `requirements.txt:1-15` zaten pinlerin `pip freeze`'den alındığını yazıyor. Codex yorum bloğunu atlamış. |

**Codex'in atladıkları:** R-01 (tracked `app.py` kırık), R-02 (`data/` gitignore), R-03 (`incident_year` sabit), R-04 (`umbrella_limit` min=-1M), R-05..R-09. Bunun bir kısmı benim prompt'u 4 dosyayla sınırlamamdan kaynaklanıyor — Codex `app.py`'ı hiç görmedi.

---

## Sonuç: Faz 0 kapatılabilir mi?

**EVET — 1 madde ön koşul olarak kapatıldıktan sonra (`R-01`).**

Bitti kriteri ("`pipeline.pkl` tek dosya, eski 2 pkl'ye bağımlılık yok") **teknik olarak karşılanmış**: tek dosya, taze process'te tek başına çalışıyor, eski pkl'ler diskte yok, eğitim byte-byte deterministik, metrikler metadata ile tutarlı, pinler gerçek.

**Faz 0 kapanmadan önce (backend-agent):**
- **R-01** — kök `app.py` artık var olmayan 3 pkl'yi yüklüyor. Ya `pipeline.pkl` kullanacak şekilde güncellenmeli ya da silinmeli. Bu doğrudan "eski pkl bağımlılığı yok" iddiasını yalanlıyor.

**Faz 1'e taşınan (blocker değil):**
- **S3** — `model.py` pipeline'ı yalnızca sabit yoldan yüklesin.
- **S4/2** — `/model-info` `metadata.json`'ın tamamını değil, seçilmiş alt kümesini döndürsün.
- **O1** — "?" → NaN dönüşümü API katmanında ele alınsın; docstring `:13-15` düzeltilsin.
- **R-05** — Pydantic `extra="forbid"`.

**Faz 2'ye taşınan:**
- **R-03** — `incident_year` sabit (2015); guardrail bunu OOD saymamalı ya da feature drop edilmeli.
- **R-04** — `umbrella_limit` min=-1M kirli satırdan geliyor.

**Nebi onayı gereken (ürün kararı, kod değil):**
- **S4** — `insured_sex` korunan özellik olarak model feature'ı ve `/model-info`'da yayımlanacak.
- **R-02** — `data/` gitignore: veri seti repoya girsin mi (yeniden üretilebilirlik) yoksa dışarıda mı kalsın?
- **R-07 / R-08** — README ve kök `requirements.txt`/`app.py` temizliği ne zaman yapılacak?
