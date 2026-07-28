import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Tailwind v4 ships its own Vite plugin: no postcss.config / tailwind.config
// files are needed, and the theme is declared with `@theme` in `src/index.css`.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // The backend's CORS default is exactly this origin (backend/app/main.py ->
    // DEFAULT_ALLOWED_ORIGINS). If the port changes, ALLOWED_ORIGINS on the
    // backend has to change too, or the browser drops the request with an
    // opaque network error.
    strictPort: true,
  },
});
