// Cross-origin isolation where the host cannot send the headers (GitHub Pages):
// on the page this file registers itself as a service worker, which adds them to
// every response, and reloads the page once when the worker takes it over.
// SharedArrayBuffer, and with it solver-threads.js, needs the isolation.
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (e) => {
    if (e.request.cache === "only-if-cached" && e.request.mode !== "same-origin") return;
    e.respondWith(
      fetch(e.request).then((response) => {
        if (response.status === 0) return response;
        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        const { status, statusText } = response;
        return new Response(response.body, { status, statusText, headers });
      }),
    );
  });
} else if (!crossOriginIsolated && isSecureContext && navigator.serviceWorker) {
  navigator.serviceWorker
    .register(document.currentScript.src)
    .then((registration) => {
      if (navigator.serviceWorker.controller) return; // isolated after all, or unsupported
      if (registration.active) location.reload();
      else navigator.serviceWorker.addEventListener("controllerchange", () => location.reload());
    })
    .catch(() => {}); // a file:// page, single-threaded
}
