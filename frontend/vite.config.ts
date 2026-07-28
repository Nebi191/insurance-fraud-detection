import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Tailwind v4 kendi Vite eklentisiyle geliyor: postcss.config / tailwind.config
// dosyalarına gerek yok, tema `src/index.css` içinde `@theme` ile tanımlanıyor.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Backend'in CORS varsayılanı tam olarak bu origin (backend/app/main.py ->
    // DEFAULT_ALLOWED_ORIGINS). Port değişirse backend'de ALLOWED_ORIGINS
    // güncellenmeli, yoksa tarayıcı isteği opak bir ağ hatasıyla düşer.
    strictPort: true,
  },
});
