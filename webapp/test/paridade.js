// Teste de paridade: o motor JS tem que dar o mesmo que o PyTorch (referencia.json).
//   node webapp/test/paridade.js
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Copia } from "../engine/copia.js";
import { Lia } from "../engine/lia.js";

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, "..", "modelo");
const manifest = JSON.parse(readFileSync(join(dir, "modelo.json"), "utf8"));
const parts = manifest.chunks.map((c) => readFileSync(join(dir, c)));
const buffer = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
let o = 0;
for (const p of parts) { buffer.set(p, o); o += p.length; }

let t0 = performance.now();
const lia = new Lia(manifest, buffer.buffer);
const { tok, model } = lia;
console.log(`carregou em ${((performance.now() - t0) / 1000).toFixed(1)} s`);
const ref = JSON.parse(readFileSync(join(here, "referencia.json"), "utf8"));
let falhas = 0;

// 1) tokenizador
let tokOk = 0;
for (const { text, ids } of ref.tokenizer) {
  const got = tok.encode(text);
  if (got.length === ids.length && got.every((v, i) => v === ids[i])) tokOk++;
  else if (falhas++ < 5) console.log("  tokenizador diferente:", JSON.stringify(text), "\n    py:", ids.join(","), "\n    js:", got.join(","));
}
console.log(`tokenizador: ${tokOk}/${ref.tokenizer.length} iguais`);

// 2) logits (prefill + 3 passos com cache)
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
let maxDiff = 0, argOk = 0, argN = 0;
t0 = performance.now();
for (const { ids, tokens, logits } of ref.logits) {
  const cache = model.novoCache();
  let lg = model.forward(ids, cache);
  for (let s = 0; s < logits.length; s++) {
    for (let i = 0; i < lg.length; i++) maxDiff = Math.max(maxDiff, Math.abs(lg[i] - logits[s][i]));
    argN++;
    if (argmax(lg) === argmax(logits[s])) argOk++;
    if (s < tokens.length) lg = model.forward([tokens[s]], cache);
  }
}
console.log(`logits: maior diferença ${maxDiff.toExponential(2)} | mesmo token mais provável ${argOk}/${argN}`);
if (maxDiff > 1e-2) falhas++;

// 3) chamadas com cópia forçada (o predict do eval.py)
function predict(userText) {
  const ids = [tok.bos, ...tok.encodeTurn("user", userText), tok.assistant];
  const cache = model.novoCache();
  let lg = model.forward(ids, cache);
  if (argmax(lg) !== tok.call) return ["assistant", null];
  const copia = new Copia(manifest.fixas, userText);
  let raw = new Uint8Array(0), t = tok.call;
  const gen = [];
  for (let s = 0; s < 48; s++) {
    lg = model.forward([t], cache);
    t = copia.escolhe(lg, raw, tok);
    if (t === null || t === tok.end) break;
    gen.push(t);
    const v = tok.vocab[t], next = new Uint8Array(raw.length + v.length);
    next.set(raw); next.set(v, raw.length); raw = next;
  }
  return ["call", tok.decode(gen).trim()];
}
let callOk = 0;
t0 = performance.now();
for (const c of ref.chamadas) {
  const [role, out] = predict(c.user);
  if (role === c.role && out === c.out) callOk++;
  else if (falhas++ < 10) console.log("  chamada diferente:", JSON.stringify(c.user), "\n    py:", c.role, c.out, "\n    js:", role, out);
}
console.log(`chamadas com cópia forçada: ${callOk}/${ref.chamadas.length} iguais ` +
  `(${((performance.now() - t0) / ref.chamadas.length).toFixed(0)} ms por pedido)`);

// 4) respostas gulosas inteiras + velocidade
let gulOk = 0, nTok = 0;
t0 = performance.now();
for (const { ids, gen } of ref.gulosas) {
  const cache = model.novoCache();
  let lg = model.forward(ids, cache);
  const out = [];
  for (let s = 0; s < 60; s++) {
    const t = argmax(lg);
    if (t === tok.end) break;
    out.push(t);
    lg = model.forward([t], cache);
    nTok++;
  }
  if (out.length === gen.length && out.every((v, i) => v === gen[i])) gulOk++;
  else if (falhas++ < 12) console.log("  resposta diferente:\n    py:", tok.decode(gen), "\n    js:", tok.decode(out));
}
const secs = (performance.now() - t0) / 1000;
console.log(`respostas gulosas: ${gulOk}/${ref.gulosas.length} iguais | ${(nTok / secs).toFixed(1)} tokens/s`);

const ok = tokOk === ref.tokenizer.length && maxDiff <= 1e-2 && callOk === ref.chamadas.length && gulOk === ref.gulosas.length;
console.log(ok ? "PARIDADE OK" : "PARIDADE COM DIFERENÇAS");
process.exit(ok ? 0 : 1);
