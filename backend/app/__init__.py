"""FastAPI servis katmanı (Faz 1).

Paket üç sorumluluğa ayrılmıştır:

    schemas.py  -> API sözleşmesi. Girdinin *fiziksel/mantıksal* geçerliliğini
                   doğrular (Pydantic). Model hakkında hiçbir şey bilmez.
    model.py    -> Artefakt yükleme, şema normalizasyonu, tahmin, SHAP.
                   HTTP hakkında hiçbir şey bilmez.
    main.py     -> İkisini birleştiren ince HTTP katmanı (endpoint + CORS).

Bu ayrımın amacı, `model.py`'ı FastAPI'siz de test edilebilir bırakmak ve
"validation nerede yapılıyor?" sorusunun tek bir cevabı olmasıdır.
"""
