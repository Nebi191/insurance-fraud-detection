/**
 * İki sayfalık bir uygulama için minik History API yönlendiricisi.
 *
 * NEDEN react-router DEĞİL
 * -----------------------
 * `react-router-dom` kurulduğunda `npm audit` iki HIGH severity açık raporladı
 * (GHSA-qwww-vcr4-c8h2, RSC modunda CSRF bypass) ve `npm audit fix` çözemedi —
 * açık kütüphanenin bütün güncel sürümlerini kapsıyor. Açık RSC moduna özgü,
 * yani SPA olarak kullanan bizi pratikte etkilemiyordu; ama bu bir portfolyo
 * demosu ve müşteri `npm audit` çıktısına bakabilir. "Etkilenmiyoruz" demek
 * yerine bağımlılığı kaldırmak hem sorunu hem 46 paketi birden çözdü.
 *
 * Kaybedilen bir şey yok: iki rota için gereken tek şey `pushState` + `popstate`
 * dinleyicisi. URL paylaşılabilirliği ve tarayıcı geri tuşu korunuyor.
 *
 * FAZ 7 NOTU: Netlify'da SPA yönlendirmesi için `_redirects` dosyası şart
 * (`/* /index.html 200`), aksi hâlde `/model-card` adresine doğrudan girildiğinde
 * 404 alınır. Bu, react-router kullansaydık da gerekliydi.
 */

import { useCallback, useEffect, useState } from "react";

export type Route = "score" | "model-card";

function routeFromPath(pathname: string): Route {
  return pathname.replace(/\/+$/, "").endsWith("/model-card") ? "model-card" : "score";
}

function pathForRoute(route: Route): string {
  // `import.meta.env.BASE_URL` sondaki eğik çizgiyi zaten içerir.
  const base = import.meta.env.BASE_URL;
  return route === "model-card" ? `${base}model-card` : base;
}

export function useRoute(): {
  route: Route;
  navigate: (next: Route) => void;
} {
  const [route, setRoute] = useState<Route>(() => routeFromPath(window.location.pathname));

  useEffect(() => {
    // Tarayıcı geri/ileri tuşu.
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
 * Gerçek bir `<a href>` — orta tık, "yeni sekmede aç" ve kopyalanabilir bağlantı
 * çalışmaya devam eder. Yalnızca düz sol tıkta sayfa yenilemesi engellenir.
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
