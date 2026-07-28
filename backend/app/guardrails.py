"""Out-of-distribution (OOD) tespiti — "model bu değeri eğitimde gördü mü?".

NE YAPAR, NEDEN BÖYLE YAPAR
---------------------------

1) NEDEN VAR
   `predict_proba` her girdi için bir sayı döndürür. Girdi eğitim verisinin
   hiç uzanmadığı bir bölgedeyse de döndürür — "bu bana yabancı" diye bir
   çıktısı YOKTUR. Skor 0.71 gelir, `risk_level` "high" yazar, SHAP grafiği
   çizilir; her şey normal görünür ama o 0.71'in arkasında hiçbir gözlem
   yoktur.

   Ağaç modellerinde bu özellikle sinsidir. LightGBM `auto_year <= 2012.5`
   gibi eşiklerle bölünür ve eğitimde 2015'ten büyük değer olmadığı için
   HİÇBİR eşik 2015'in üstünde olamaz. Sonuç: 2016 model araç ile 2050 model
   araç aynı yaprağa düşer, aynı katkıyı alır. Model "çok yeni araç" ayrımını
   yapamaz ama yaptığını sanırsınız.

   Bu modül o sessizliği sese çevirir. Tahmini ENGELLEMEZ — skor yine döner,
   yanına "şu alanları daha önce hiç görmedim" notu eklenir. Sigorta eksperi
   için bu kritik bir fark: reddedilen bir talep ile modelin emin olmadığı bir
   talep aynı şey değildir.

2) YALNIZCA SAYISAL ALANLAR — VE BU BİR EKSİKLİK DEĞİL
   Kategorik alanlar `schemas.py`'da `Literal` ile eğitimde görülen değerlere
   kapalıdır: geçersiz bir kategori guardrail'e HİÇ ULAŞMADAN 422 alır. Yani
   kategorik OOD kontrolü yazsaydık ÖLÜ KOD olurdu — API üzerinden asla
   tetiklenemeyen, dolayısıyla gerçekten test edilemeyen bir dal.

   Bu bilinçli bir tasarım: `OrdinalEncoder` `unknown_value=-1` ile kurulu,
   bilinmeyen kategoriyi sessizce -1'e kodlayıp anlamsız ama "başarılı"
   görünen bir tahmin üretirdi. 422 dönmek daha güvenli. Ama sınırın kendisi
   saklanmamalı: `/predict` yanıtındaki alan açıklaması ve model card bunu
   açıkça söyler.

3) YALNIZCA GÖNDERİLEN ALANLAR KONTROL EDİLİR
   Verilmeyen alanlar `metadata.defaults`'taki medyan/mod ile doldurulur ve o
   değerler TANIM GEREĞİ eğitim aralığının içindedir. Doldurulmuş bir alan için
   uyarı üretmek, kullanıcının hiç dokunmadığı bir alanı sorunluymuş gibi
   göstermek olurdu — üstelik her boş istek 34 uyarı basardı.

   Ayrım `None` üzerinden yapılır: alan hiç gönderilmedi ya da açıkça `null`
   gönderildi -> kontrol yok. Bu, `prepare_row()`'un varsayılan doldurma
   kuralıyla birebir aynı ayrımdır (bkz. `model.py` K3/K4).

4) SIRALAMA ANLAMLIDIR: ÖNCE MODELİN GERÇEKTEN BAKTIĞI ALANLAR
   Bu modelin 34 feature'ından 16'sının split sayısı sıfırdır — model onlara
   hiç bakmaz, SHAP katkıları tam 0.0'dır. `umbrella_limit` (ölü) için aralık
   dışı bir değer geldiğinde skor GERÇEKTEN etkilenmez; `age` (canlı) için
   geldiğinde etkilenir. İkisini aynı sırada listelemek, okuyucuyu ikisinin
   aynı ağırlıkta olduğuna inandırır.

   Uyarı yine de ÜRETİLİR (kullanıcı "aralık dışı girdim, sistem fark etti mi?"
   sorusunun cevabını hak eder), ama etkili olanlar listenin başına alınır.

   API sözleşmesi (CLAUDE.md) bu alanı düz bir string listesi olarak tanımlıyor
   ve öyle kalıyor — etkinlik bilgisi zaten `/model-info -> feature_influence`
   üzerinden servis ediliyor, frontend iki listeyi eşleştirebilir. Sözleşmeyi
   genişletmek yerine sıralamayı anlamlı kılmak, aynı bilgiyi kırmadan taşır.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

# Sözleşmede sayısal kabul ettiğimiz tipler. `bool` BİLEREK dışarıda:
# Python'da `bool` `int`in alt sınıfıdır, yani `isinstance(True, int)` doğrudur
# ve `True` sessizce 1 gibi karşılaştırılırdı. Sayısal bir alana `true`
# gönderen bir istemci hata değil "1" muamelesi görürdü.
NUMERIC_TYPES = (int, float)


class Guardrail:
    """Eğitim aralığı dışında kalan SAYISAL alanları tespit eder.

    Metadata'dan bir kez kurulur (uygulama açılışında), sonra her istekte
    `check()` çağrılır. Aralıklar KODA GÖMÜLMEZ — `metadata.training_ranges`'ten
    okunur, böylece model yeniden eğitilince guardrail kendiliğinden güncellenir.
    """

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        training_ranges: Mapping[str, Mapping[str, Any]] = metadata["training_ranges"]
        influence: Mapping[str, Mapping[str, Any]] = metadata["feature_influence"]["features"]
        input_order: Sequence[str] = metadata["feature_list"]["pipeline_input_order"]

        # {alan: (min, max)} — yalnızca sayısal alanlar.
        self.numeric_bounds: dict[str, tuple[float, float]] = {
            name: (float(spec["min"]), float(spec["max"]))
            for name, spec in training_ranges.items()
            if spec["type"] == "numeric"
        }

        # Sıralama anahtarı: (etkisiz mi?, girdi sırasındaki konum).
        # `False < True` olduğu için etkili alanlar önce gelir; ikinci bileşen
        # aynı grup içinde deterministik ve okunabilir bir sıra verir.
        position = {name: index for index, name in enumerate(input_order)}
        self._sort_key: dict[str, tuple[bool, int]] = {
            name: (not influence[name]["has_influence"], position[name])
            for name in self.numeric_bounds
        }

    def check(self, payload: Mapping[str, Any]) -> list[str]:
        """Gönderilen alanlardan eğitim aralığı dışında kalanların adları.

        Dönen liste: önce modelin gerçekten kullandığı alanlar, sonra ölü
        alanlar; her grup içinde pipeline girdi sırası.
        """
        flagged: list[str] = []

        for name, (minimum, maximum) in self.numeric_bounds.items():
            value = payload.get(name)
            if value is None:
                # Alan gönderilmedi -> varsayılanla doldurulacak (bkz. 3).
                continue
            if isinstance(value, bool) or not isinstance(value, NUMERIC_TYPES):
                # Sayısal olmayan değer bu katmanın işi değil: Pydantic zaten
                # 422 döndürür. Guardrail doğrudan (HTTP'siz) çağrıldığında da
                # sessizce yanlış karşılaştırma yapmaktansa atlamak doğru.
                continue

            try:
                numeric = float(value)
            except OverflowError:
                # `10**10000` gibi devasa bir `int` float'a sığmaz (Codex F2-3).
                # HTTP yolundan erişilemez (Pydantic sınırları çok daha dar ve
                # JSON kodlayıcı zaten patlar), ama bu modül HTTP'siz de
                # çağrılabiliyor. Aralığın dışında olduğu kesin: işaretle, patlama.
                flagged.append(name)
                continue

            if math.isnan(numeric):
                # NaN her karşılaştırmada False verir; kontrol etmeseydik
                # "aralık içinde" sayılıp SESSİZCE geçerdi.
                flagged.append(name)
                continue

            if numeric < minimum or numeric > maximum:
                flagged.append(name)

        flagged.sort(key=lambda name: self._sort_key[name])
        return flagged
