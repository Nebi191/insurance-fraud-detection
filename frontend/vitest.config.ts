import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Separate from `vite.config.ts` on purpose: the dev/build config wires up the
// Tailwind Vite plugin and a fixed dev-server port, neither of which a jsdom
// test run needs or benefits from. Sharing the React plugin is enough to get
// JSX/Fast-Refresh-free transforms right in tests.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/setupTests.ts"],
    css: false,
  },
});
