/**
 * A tiny History API router for a two-page application.
 *
 * WHY NOT react-router
 * --------------------
 * Installing `react-router-dom` made `npm audit` report two HIGH severity
 * advisories (GHSA-qwww-vcr4-c8h2, a CSRF bypass in RSC mode) and `npm audit fix`
 * could not resolve them — the advisory covers every current release of the
 * library. The vulnerability is specific to RSC mode, so using it as an SPA did
 * not put us at practical risk; but this is a portfolio demo and a client may
 * well look at `npm audit` output. Dropping the dependency, rather than arguing
 * "we are not affected", resolved both the finding and 46 packages at once.
 *
 * Nothing was lost: two routes need nothing more than `pushState` plus a
 * `popstate` listener. Shareable URLs and the browser back button still work.
 *
 * PHASE 7 NOTE: Netlify needs a `_redirects` file for SPA routing
 * (`/* /index.html 200`), otherwise navigating straight to `/model-card` returns
 * a 404. That would have been required with react-router too.
 */

import { useCallback, useEffect, useState } from "react";

export type Route = "score" | "model-card";

function routeFromPath(pathname: string): Route {
  return pathname.replace(/\/+$/, "").endsWith("/model-card") ? "model-card" : "score";
}

function pathForRoute(route: Route): string {
  // `import.meta.env.BASE_URL` already includes the trailing slash.
  const base = import.meta.env.BASE_URL;
  return route === "model-card" ? `${base}model-card` : base;
}

export function useRoute(): {
  route: Route;
  navigate: (next: Route) => void;
} {
  const [route, setRoute] = useState<Route>(() => routeFromPath(window.location.pathname));

  useEffect(() => {
    // Browser back/forward buttons.
    const onPopState = () => setRoute(routeFromPath(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((next: Route) => {
    window.history.pushState({}, "", pathForRoute(next));
    setRoute(next);
    window.scrollTo({ top: 0 });
  }, []);

  return { route, navigate };
}

/**
 * A real `<a href>` — middle click, "open in new tab" and copying the link all
 * keep working. Only a plain left click is intercepted to avoid a full reload.
 */
export function RouteLink({
  to,
  navigate,
  className,
  children,
}: {
  to: Route;
  navigate: (next: Route) => void;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <a
      href={pathForRoute(to)}
      className={className}
      onClick={(event) => {
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
        event.preventDefault();
        navigate(to);
      }}
    >
      {children}
    </a>
  );
}
