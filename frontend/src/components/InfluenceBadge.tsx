/**
 * "Model bu alanı kullanmıyor" rozeti.
 *
 * NEDEN VAR
 * ---------
 * Bu modelin 34 feature'ından 16'sının booster'daki split sayısı SIFIR: model
 * o kolonlarda hiç dallanmıyor, dolayısıyla SHAP katkıları tam 0.0 ve alanın
 * değerini değiştirmek skoru bit düzeyinde bile oynatmıyor (backend'de
 * `test_dead_features_change_neither_the_score_nor_their_own_shap_value` ile
 * 40'tan fazla değer kombinasyonunda kanıtlanıyor).
 *
 * Bu alanları formdan GİZLEMEK modelin sınırlılığını görünmez yapardı. Rozet,
 * zayıflığı şeffaflığa çeviriyor: "API'm hangi girdinin boşuna sorulduğunu da
 * söylüyor."
 *
 * VERİ KAYNAĞI: `/model-info -> feature_influence`. Liste koda GÖMÜLÜ DEĞİL;
 * model yeniden eğitilirse rozetler kendiliğinden güncellenir.
 */

export function InfluenceBadge({ splitCount }: { splitCount: number }) {
  return (
    <span
      className="group/badge relative inline-flex shrink-0 items-center gap-1 rounded-full border border-slate-300 bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400"
      // Klavye ile de odaklanabilsin ki açıklama fare olmadan da okunabilsin.
      tabIndex={0}
    >
      <svg viewBox="0 0 24 24" className="size-3" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <path d="M6 6l12 12" strokeLinecap="round" />
      </svg>
      model kullanmıyor
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-0 z-20 mb-1.5 hidden w-64 rounded-lg border border-slate-300 bg-white p-2.5 text-[11px] leading-relaxed font-normal text-slate-700 shadow-lg group-hover/badge:block group-focus-within/badge:block dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
      >
        Eğitilmiş ağaçlar bu kolonda hiç dallanma yapmadı ({splitCount} bölünme).
        Değerini değiştirmek skoru <strong>hiç etkilemez</strong> ve SHAP katkısı
        her zaman tam 0,0 olur.
        <span className="mt-1.5 block text-slate-500 dark:text-slate-400">
          Bu, alanın önemsiz olduğu anlamına gelmez — yalnızca <em>bu</em> eğitilmiş
          modelin ona bakmadığını gösterir.
        </span>
      </span>
    </span>
  );
}
