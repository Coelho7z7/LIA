// Service worker da Lia na web: guarda a "casca" do site para abrir rápido e funcionar sem internet.
// Os pesos do modelo ficam num cache à parte, cuidado pelo engine/worker.js.
const SHELL = "lia-shell-__BUILD__";
const FILES = ["/", "/backend.js", "/ferramentas.js", "/engine/lia.js", "/engine/model.js", "/engine/tokenizer.js",
  "/engine/copia.js", "/engine/worker.js", "/manifest.json", "/icon-192.png", "/icon-512.png", "/privacidade.html",
  "/modelo/modelo.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((ks) => Promise.all(ks.filter((k) => k.startsWith("lia-shell-") && k !== SHELL).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  // pesos (.bin) e outros sites seguem direto: o worker do modelo tem o cache dele
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.endsWith(".bin")) return;
  // rede primeiro (pega a versão nova); sem internet, a guardada
  e.respondWith(fetch(e.request)
    .then((r) => {
      if (r.ok) { const copia = r.clone(); caches.open(SHELL).then((c) => c.put(e.request, copia)); }
      return r;
    })
    .catch(() => caches.match(e.request).then((r) => r || caches.match("/"))));
});
