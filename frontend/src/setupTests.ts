/**
 * Global Vitest setup: registers the jest-dom matchers (`toBeInTheDocument`,
 * `toHaveTextContent`, ...) on Vitest's own `expect`, and cleans up the jsdom
 * DOM between tests so one test's rendered tree cannot leak into the next.
 */
import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
