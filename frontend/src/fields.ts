/**
 * Form alanlarının İNSAN KARARI olan kısmı: Türkçe etiketler, gruplama, sıra.
 *
 * NEDEN YARI-DİNAMİK
 * ------------------
 * Burada BULUNMAYAN her şey `/model-info`'dan gelir: seçenek listeleri,
 * min/max aralıkları, varsayılan değerler ve —en önemlisi— hangi alanın modeli
 * etkilemediği (`feature_influence`).
 *
 * Etkisizlik listesini buraya gömmek YASAK: model yeniden eğitilince liste
 * değişir ve gömülü kopya sessizce yalan söylemeye başlar. Rozetler tek
 * doğruluk kaynağından (artefakt) beslenir.
 *
 * Etiket ve gruplama ise API'den gelemez — bunlar çeviri ve bilgi mimarisi
 * kararlarıdır. Alan adı artefaktta değişirse `assertFieldsCoverContract()`
 * bunu açılışta yakalar; sessizce eksik form göstermeyiz.
 */

export interface FieldMeta {
  /** API alan adı — `capital-gains` gibi tireli hâliyle. */
  name: string;
  label: string;
  /** Etiketin altında görünen kısa açıklama. */
  hint?: string;
  /** Sayısal alanlarda girdi adımı (ondalıklı primler için 0.01). */
  step?: number;
}

export interface FieldGroup {
  id: string;
  title: string;
  description: string;
  fields: FieldMeta[];
}

/**
 * Accordion'da AÇIK başlayan grup: skoru fiilen sürükleyen alanlar.
 *
 * Seçim ölçüme dayanıyor (booster split sayıları): insured_hobbies 353,
 * incident_severity 173, policy_annual_premium 107, capital-gains 75,
 * auto_year 74, witnesses 37. Liste burada sabit çünkü hangi alanların
 * "vitrine" çıkacağı bir sunum kararı; rozetler ise ölçümden gelir.
 */
export const HIGHLIGHT_GROUP: FieldGroup = {
  id: "highlights",
  title: "Skoru en çok etkileyenler",
  description:
    "Modelin en sık dallandığı altı alan. Diğer alanları boş bırakırsanız " +
    "eğitim verisinin medyan/mod değerleriyle doldurulur.",
  fields: [
    { name: "insured_hobbies", label: "Hobi" },
    { name: "incident_severity", label: "Hasar şiddeti" },
    { name: "policy_annual_premium", label: "Yıllık prim", step: 0.01 },
    { name: "capital-gains", label: "Sermaye kazancı" },
    { name: "auto_year", label: "Araç model yılı" },
    { name: "witnesses", label: "Tanık sayısı" },
  ],
};

/** Kalan alanlar, mantıksal gruplarında. */
export const FIELD_GROUPS: FieldGroup[] = [
  {
    id: "policy",
    title: "Poliçe / müşteri",
    description: "Sigorta ilişkisine ve poliçe koşullarına ait alanlar.",
    fields: [
      { name: "months_as_customer", label: "Müşteri süresi", hint: "Ay cinsinden" },
      { name: "age", label: "Yaş" },
      { name: "policy_state", label: "Poliçe eyaleti" },
      { name: "policy_csl", label: "Teminat limiti (CSL)", hint: "Kişi başı / olay başı" },
      { name: "policy_deductable", label: "Muafiyet" },
      { name: "umbrella_limit", label: "Şemsiye poliçe limiti" },
      { name: "policy_bind_year", label: "Poliçe başlangıç yılı" },
    ],
  },
  {
    id: "insured",
    title: "Sigortalı profili",
    description: "Sigortalının demografik ve finansal bilgileri.",
    fields: [
      { name: "insured_sex", label: "Cinsiyet" },
      { name: "insured_education_level", label: "Eğitim düzeyi" },
      { name: "insured_occupation", label: "Meslek" },
      { name: "insured_relationship", label: "Hane ilişkisi" },
      {
        name: "capital-loss",
        label: "Sermaye zararı",
        hint: "Bu veri setinde negatif değerle kodlanır",
      },
    ],
  },
  {
    id: "incident",
    title: "Olay",
    description: "Kazanın kendisine ait alanlar.",
    fields: [
      { name: "incident_type", label: "Olay tipi" },
      { name: "collision_type", label: "Çarpışma tipi" },
      { name: "authorities_contacted", label: "Bilgilendirilen birim" },
      { name: "incident_state", label: "Olay eyaleti" },
      { name: "incident_city", label: "Olay şehri" },
      { name: "incident_hour_of_the_day", label: "Olay saati", hint: "0-23" },
      { name: "number_of_vehicles_involved", label: "Karışan araç sayısı" },
      { name: "property_damage", label: "Mal hasarı var mı" },
      { name: "bodily_injuries", label: "Yaralı sayısı" },
      { name: "police_report_available", label: "Polis raporu var mı" },
      {
        name: "incident_year",
        label: "Olay yılı",
        hint: "Eğitim verisinin tamamı 2015 — başka bir yıl dağılım dışı sayılır",
      },
    ],
  },
  {
    id: "claim",
    title: "Tazminat kalemleri",
    description: "Talep edilen tutarlar.",
    fields: [
      { name: "total_claim_amount", label: "Toplam talep tutarı" },
      { name: "injury_claim", label: "Yaralanma tazminatı" },
      { name: "property_claim", label: "Mal tazminatı" },
      { name: "vehicle_claim", label: "Araç tazminatı" },
    ],
  },
  {
    id: "vehicle",
    title: "Araç",
    description: "Araca ait alanlar.",
    fields: [{ name: "auto_make", label: "Araç markası" }],
  },
];

export const ALL_GROUPS: FieldGroup[] = [HIGHLIGHT_GROUP, ...FIELD_GROUPS];

/** Form genelinde tanımlı bütün alan adları. */
export const ALL_FIELD_NAMES: string[] = ALL_GROUPS.flatMap((group) =>
  group.fields.map((field) => field.name),
);

/**
 * API alan adı -> Türkçe etiket.
 *
 * SHAP grafiği ve uyarı listeleri backend'den ham alan adlarıyla geliyor
 * (`capital-gains` gibi); kullanıcıya gösterilirken aynı etiketi kullanmak
 * formla grafiği birbirine bağlar. Tablo `ALL_GROUPS`'tan TÜRETİLİYOR — ikinci
 * bir elle liste tutmak, ikisinin ayrışması demek olurdu.
 */
export const FIELD_LABELS: Record<string, string> = Object.fromEntries(
  ALL_GROUPS.flatMap((group) => group.fields.map((field) => [field.name, field.label])),
);

/** Etiket bulunamazsa ham adı döndürür — sessizce boş göstermez. */
export function fieldLabel(name: string): string {
  return FIELD_LABELS[name] ?? name;
}

/**
 * Kategorik değerlerin okunabilir karşılıkları.
 *
 * API'ye GÖNDERİLEN değer her zaman anahtarın kendisidir; burada yalnızca
 * gösterim değişir. Bütün kategorileri çevirmiyoruz (hobi, meslek, marka gibi
 * alanlar veri setinin ham değerleriyle daha dürüst okunuyor) — yalnızca
 * anlamı Türkçede kaybolanları.
 */
export const VALUE_LABELS: Record<string, string> = {
  "?": "Bilinmiyor (?)",
  YES: "Evet",
  NO: "Hayır",
  MALE: "Erkek",
  FEMALE: "Kadın",
  "Major Damage": "Ağır hasar",
  "Minor Damage": "Hafif hasar",
  "Total Loss": "Pert (tam hasar)",
  "Trivial Damage": "Çok hafif hasar",
};

export function valueLabel(value: string): string {
  return VALUE_LABELS[value] ?? value;
}

/**
 * Formun API sözleşmesini tam kapsadığını doğrular.
 *
 * NEDEN GEREKLİ: alan listesi elle yazıldığı için artefakt değiştiğinde
 * sessizce eskir. Eksik alan = kullanıcının hiç göremediği bir girdi; fazla
 * alan = `extra="forbid"` yüzünden her isteğin 422 alması. İkisi de sessiz
 * değil, gürültülü başarısız olmalı.
 */
export function assertFieldsCoverContract(pipelineInputOrder: string[]): void {
  const declared = new Set(ALL_FIELD_NAMES);
  const contract = new Set(pipelineInputOrder);

  const missing = pipelineInputOrder.filter((name) => !declared.has(name));
  const extra = ALL_FIELD_NAMES.filter((name) => !contract.has(name));
  const duplicates = ALL_FIELD_NAMES.filter(
    (name, index) => ALL_FIELD_NAMES.indexOf(name) !== index,
  );

  const problems: string[] = [];
  if (missing.length) problems.push(`formda olmayan alan(lar): ${missing.join(", ")}`);
  if (extra.length) problems.push(`API'de olmayan alan(lar): ${extra.join(", ")}`);
  if (duplicates.length) problems.push(`iki kez tanımlı: ${duplicates.join(", ")}`);

  if (problems.length) {
    throw new Error(
      `Form alanları API sözleşmesiyle uyuşmuyor — ${problems.join(" | ")}. ` +
        "src/fields.ts güncellenmeli.",
    );
  }
}
