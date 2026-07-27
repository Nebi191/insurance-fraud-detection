# Insurance Fraud Detection — Profesyonel Demo Projesi

## Amaç
Mevcut `insurance-fraud-detection` reposundaki (Nebi191, LightGBM + Optuna,
Test PR-AUC ~0.65) ML pipeline'ın üzerine, Upwork müşterilerine (sigorta /
fintech risk değerlendirme) gösterilecek, production-kalitesinde bir demo
inşa etmek. Bu bir Kaggle-notebook demosu DEĞİL, "bu adam bana ürün teslim
edebilir" demosu olmalı. İkincil amaç: sürecin her adımı öğretici — kopyala-
yapıştır değil, her kararın gerekçesi anlaşılır olmalı.

---

## Kesin Teknoloji Kararları (değiştirme, tartışma yok)

- **Backend:** FastAPI + Pydantic v2, Python 3.11+
- **Model paketleme:** Mevcut `preprocessor.pkl` + `best_model_lgb.pkl` TEK bir
  sklearn `Pipeline` içinde birleştirilecek (`Pipeline([('preprocessor', ...),
  ('model', ...)])`, tek `.pkl` olarak kaydedilir). Elle/manuel encoding
  sözlüğü YASAK — preprocessing her zaman pipeline üzerinden yürür.
- **Explainability:** SHAP waterfall matplotlib PNG olarak DEĞİL, backend'den
  ham JSON (`feature`, `shap_value`, `base_value`) olarak gönderilir, frontend'de
  interaktif çizilir.
- **Guardrail:** `guardrails.py` — training verisinde gözlemlenmemiş
  (min/max dışı) input değerlerinde sessizce ekstrapolasyon yapmak yerine
  `out_of_distribution_warnings` alanında uyarı döner. (Coffee Intelligence
  Module 2'de karşılaştığın sorunun aynısı, bu sefer baştan çözülüyor.)
- **Frontend:** React + Vite + TypeScript + Tailwind
- **Grafik:** recharts (SHAP JSON'unu interaktif waterfall/bar olarak çizer)
- **Deploy:** Backend → Hugging Face Spaces (Docker), Frontend → Netlify
- **Repo yapısı:** monorepo, `backend/` ve `frontend/` ayrı klasörler
  (Netlify sadece frontend'i host edebiliyor, backend ayrı deploy edilir)

---

## Repo Yapısı

```
insurance-fraud-detection/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, endpoint'ler
│   │   ├── schemas.py       # Pydantic request/response modelleri
│   │   ├── model.py         # pipeline yükleme, predict, SHAP
│   │   └── guardrails.py    # OOD / extrapolasyon kontrolü
│   ├── models/
│   │   └── pipeline.pkl     # preprocessor + model birleşik
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   └── package.json
├── notebooks/                # mevcut 01-05, artık "keşif" amaçlı
├── data/
├── review_log/                # Codex CLI raporları, tarihli
├── .claude/agents/
│   ├── backend-agent.md
│   ├── frontend-agent.md
│   └── reviewer.md
└── README.md
```

---

## API Sözleşmesi

**POST /predict**
```json
// Request
{
  "incident_severity": "Major Damage",
  "incident_type": "Single Vehicle Collision",
  "collision_type": "Side Collision",
  "auto_year": 2010,
  "number_of_vehicles_involved": 1,
  "witnesses": 2,
  "police_report_available": "YES",
  "capital-gains": 30000,
  "capital-loss": 0,
  "total_claim_amount": 55000
}

// Response
{
  "fraud_probability": 0.71,
  "risk_level": "high",
  "shap_values": [
    {"feature": "incident_severity", "value": 1.83, "base_value": -0.4}
  ],
  "out_of_distribution_warnings": ["capital-gains"]
}
```
Verilmeyen alanlar `defaults.pkl`'deki medyan/mod değerleriyle doldurulur
(mevcut mantık zaten bunu yapıyor, backend'e taşınacak).

**GET /model-info** → PR-AUC, eğitim tarihi, feature listesi (model card
sayfası için)

**GET /health** → basit healthcheck

---

## Faz Listesi + Bitti Kriteri

| Faz | İçerik | Agent | Bitti Kriteri |
|---|---|---|---|
| 0 | Pipeline konsolidasyonu (preprocessor+model tek pkl) | backend-agent | `pipeline.pkl` tek dosya, eski 2 pkl'ye bağımlılık yok |
| 1 | FastAPI endpoint'leri + Pydantic validation | backend-agent | `/predict`, `/health`, `/model-info` çalışıyor, pytest yeşil |
| 2 | Guardrail sistemi | backend-agent | Training dışı değerde `out_of_distribution_warnings` doğru tetikleniyor, testi var |
| 3 | React iskeleti + form + sonuç kartı | frontend-agent | API'ye gerçek istek atıyor, sonucu gösteriyor |
| 4 | SHAP interaktif chart + OOD uyarı banner'ı | frontend-agent | Statik PNG yok, recharts ile canlı render |
| 5 | Model-info / model card sayfası | frontend-agent | PR-AUC, versiyon, feature listesi görünür |
| 6 | Reviewer turu (Codex CLI ilk devreye giriş) | reviewer | review_log'a rapor düştü, bug/security maddeleri kapatıldı |
| 7 | Deploy (HF Spaces + Netlify) + CORS | backend-agent | Public URL'ler çalışıyor, cross-origin hatasız |

---

## Subagent Mimarisi

| Agent | Görev | Araç Yetkisi | Model |
|---|---|---|---|
| `backend-agent` | FastAPI, pipeline, guardrail, API sözleşmesi | Read, Write, Edit, Bash, Glob, Grep | Sonnet 5 |
| `frontend-agent` | React/Vite/UI, SHAP görselleştirme | Read, Write, Edit, Bash, Glob, Grep | Sonnet 5 |
| `reviewer` | İç test/lint + Codex CLI ikinci görüş, kod YAZMAZ | Read, Bash, Grep, Glob | Sonnet 5 |

Mimari kararlar ve deploy gate'lerinde (Faz 0, Faz 7) Opus 4.8 + `/effort max`
kullan. Tek seferlik derin debug/HPO takıntısında `ultrathink`. Fable 5 sadece
kesinlikle gerekirse.

**Kritik kural:** `reviewer` bulduğu sorunu kendisi düzeltmez — bulguyu
raporlar, düzeltmeyi `backend-agent`/`frontend-agent` yapar. UI/UX bulgularında
ilgili agent kod yazmadan önce sana 2-3 somut seçenek sunup onay bekler; bug ve
security bulgularını doğrudan düzeltebilir.

---

## Bilinen Hata Checklist (her fazın sonunda kontrol et)

- [ ] Optuna `TPESampler(seed=...)` sabitlendi mi?
- [ ] Train/test split, feature engineering'den ÖNCE mi yapıldı?
- [ ] Elle encoding sözlüğüne geri dönülmedi mi (tek pipeline kuralı korunuyor mu)?
- [ ] SHAP versiyonu `requirements.txt`'de sabit mi?
- [ ] Guardrail gerçekten "unseen" değerde tetikleniyor mu (test var mı, varsayılmadı mı)?
- [ ] `/predict` response'unda hassas/gereksiz alan sızdırılmıyor mu (security)?

---

## Notlar
- 21st.dev henüz derinlemesine kullanılmadı — `frontend-agent` bunu opsiyonel
  bir bileşen kaynağı olarak değerlendirebilir, zorunlu değil.
- Codex CLI (ChatGPT) `AGENTS.md` konvansiyonunu okur ama burada ayrı bir
  `AGENTS.md` tutmuyoruz — context, `reviewer` çağrısında inline veriliyor.
