// ESLint flat config for the frontend.
//
// Type-aware linting is enabled on purpose (`projectService`): the rules that
// actually catch bugs in this codebase — floating promises around the fetch
// layer, misused `await`, unsafe narrowing of the API response types — all need
// the type checker. Syntax-only linting would duplicate what `tsc -b` already
// does in `npm run build`.
import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules"] },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommendedTypeChecked,
      // `configs["recommended-latest"]` is the legacy eslintrc shape (plugins as
      // an array); the flat-config equivalent lives under `configs.flat`.
      reactHooks.configs.flat["recommended-latest"],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  // This config file is not listed in tsconfig.json's `include`, so the type
  // checker has no program for it and type-aware rules cannot run.
  // `vite.config.ts` IS included, so it stays fully checked.
  {
    files: ["eslint.config.js"],
    extends: [tseslint.configs.disableTypeChecked],
  },
);
