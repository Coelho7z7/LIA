// Service worker do app da Lia: só cuida da abertura da página.
// Tudo o mais (API, conversa) vai direto para o servidor local, sem cache.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

const OFFLINE = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lia</title>
<style>body{margin:0;height:100vh;display:grid;place-items:center;background:#121214;color:#ecebe8;
font:15px/1.6 "Segoe UI",system-ui,sans-serif;text-align:center}.b{max-width:360px;padding:24px}
.l{width:64px;height:64px;border-radius:16px;background:#f2f1ee;margin:0 auto 18px;display:grid;place-items:center}
h2{margin:0 0 6px;font-size:22px}p{color:#9b98a3;margin:0}code{background:#232328;padding:2px 6px;border-radius:6px}</style>
<div class="b"><div class="l"><svg width="40" height="40" viewBox="0 0 64 64" fill="#111"><rect x="18" y="14" width="10" height="36"/>
<rect x="18" y="40" width="28" height="10"/><rect x="36" y="24" width="10" height="10"/></svg></div>
<h2>A Lia está desligada</h2><p>Abra o atalho <b>Lia</b> no Menu Iniciar (ou rode <code>python web.py</code>) e recarregue esta janela.</p></div>`;

self.addEventListener("fetch", (e) => {
  if (e.request.mode !== "navigate") return;  // só a abertura da página; o resto segue normal
  e.respondWith(fetch(e.request).catch(() => new Response(OFFLINE, { headers: { "Content-Type": "text/html; charset=utf-8" } })));
});
