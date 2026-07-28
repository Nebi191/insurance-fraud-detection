/// <reference types="vite/client" />

// `import.meta.env` tipleri ve `.css` gibi varlık importlarının modül
// bildirimleri bu referanstan gelir.

interface ImportMetaEnv {
  /**
   * Backend adresi. Faz 7'de Netlify ortam değişkeni olarak HF Spaces URL'i
   * verilecek; yerelde tanımsız bırakılıp varsayılan kullanılır.
   */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
