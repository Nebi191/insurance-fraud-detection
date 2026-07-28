/// <reference types="vite/client" />

// This reference supplies the `import.meta.env` types and the module
// declarations for asset imports such as `.css`.

interface ImportMetaEnv {
  /**
   * Backend address. In Phase 7 Netlify supplies the Hugging Face Spaces URL as
   * an environment variable; locally it is left undefined and the default
   * applies.
   */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
