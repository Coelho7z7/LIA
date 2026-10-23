// Roda a Lia num Web Worker: o modelo pensa sem travar a página.
// Cada pedido leva um id; as respostas voltam com o mesmo id e terminam em {id, type:"fim", ...}.
//   {type:"carregar", base, opts}   baixa (ou pega do cache) os pesos -> {type:"progresso"}... {type:"pronta", info}
//   {type:"falar", text}            fala da pessoa + uma fala da Lia (eventos token/text/call)
//   {type:"resultado", text}        resultado da ferramenta pendente + uma fala da Lia
//   {type:"retry"}                  tira a última resposta em texto e gera outra (fim.ok = false se não dá)
//   {type:"anexar", turns}          só coloca falas no histórico ([papel, texto], sem gerar nada)
//   {type:"load", turns}            troca o histórico inteiro ({role, text}) — reabrir conversa salva
//   {type:"turns"}                  devolve o histórico em fim.turns
//   {type:"reset"} {type:"config", temperature}
import { Lia } from "./lia.js";

let lia = null;

async function baixar(url, cache, onBytes) {
  const hit = cache && (await cache.match(url));
  if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onBytes(b.length); return b; }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const reader = r.body.getReader(), parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); n += value.length; onBytes(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  if (cache) { try { await cache.put(url, new Response(out)); } catch { /* sem espaço: segue sem cache */ } }
  return out;
}

async function carregar(base, opts) {
  const manifest = await (await fetch(`${base}/modelo.json`, { cache: "no-cache" })).json();
  // guarda os pesos no navegador: da segunda vez abre sem baixar; modelo novo = cache novo
  const nome = `lia-pesos-${manifest.chunks[0]}`;
  let cache = null;
  try {
    cache = await caches.open(nome);
    for (const k of await caches.keys()) if (k.startsWith("lia-pesos-") && k !== nome) await caches.delete(k);
  } catch { cache = null; }
  const buffer = new Uint8Array(manifest.bytes);
  let feito = 0, o = 0;
  for (const c of manifest.chunks) {
    const part = await baixar(`${base}/${c}`, cache, (n) => {
      feito += n;
      postMessage({ type: "progresso", feito, total: manifest.bytes });
    });
    buffer.set(part, o); o += part.length;
  }
  const t0 = performance.now();
  lia = new Lia(manifest, buffer.buffer, opts);
  return { ...lia.info, ms: Math.round(performance.now() - t0) };
}

function rodar(id) {
  for (const ev of lia.step()) postMessage({ id, ...ev });
}

self.onmessage = async (e) => {
  const { id, type, ...m } = e.data;
  try {
    let extra = {};
    if (type === "carregar") { postMessage({ id, type: "pronta", info: await carregar(m.base, m.opts) }); }
    else if (type === "falar") { lia.addUser(m.text); rodar(id); }
    else if (type === "resultado") { lia.resolve(m.text); rodar(id); }
    else if (type === "retry") {
      const t = lia.turns;
      if (lia.pending || !t.length || t[t.length - 1][0] !== "assistant") extra.ok = false;
      else { t.pop(); rodar(id); extra.ok = true; }
    } else if (type === "anexar") {
      for (const [role, text] of m.turns) lia.turns.push([role, text, lia.tok.encodeTurn(role, text)]);
      lia.pending = null;
    } else if (type === "load") {
      lia.reset();
      for (const t of (m.turns || []).slice(-200)) {
        const role = t && t.role, text = String((t && t.text) || "").slice(0, 2000);
        if (["user", "assistant", "call", "result"].includes(role)) lia.turns.push([role, text, lia.tok.encodeTurn(role, text)]);
      }
      if (lia.turns.length && lia.turns[lia.turns.length - 1][0] === "call") lia.turns.pop();  // confirmação que ficou pendente
    } else if (type === "turns") extra.turns = lia.turns.map(([role, text]) => ({ role, text }));
    else if (type === "reset") lia.reset();
    else if (type === "config") { if (typeof m.temperature === "number") lia.temperature = m.temperature; }
    if (type !== "carregar") postMessage({ id, type: "fim", ...extra });
  } catch (err) {
    postMessage({ id, type: "erro", message: String((err && err.message) || err) });
  }
};
