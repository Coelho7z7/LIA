// O "cérebro" da Lia no navegador: o mesmo laço do agent.py (modelo + tokenizador + cópia forçada).
// As ferramentas ficam de fora: step() avisa da chamada e quem usa a Lia executa e chama resolve().

import { Copia } from "./copia.js";
import { Modelo } from "./model.js";
import { Tokenizer } from "./tokenizer.js";

export const FALLBACK_TEXT = "Hmm, me enrolei aqui. Fala de outro jeito?";
// Ferramentas cujo resultado é um VALOR que a resposta tem que repetir certinho.
export const FAITHFUL = {
  calcular: "Dá {v}.", hora: "São {v}.", data: "Hoje é {v}.", clima: "Olha só: {v}.",
  recordar: "Pelo que você me contou: {v}.", ver_notas: "Suas últimas notas: {v}.",
  consultar: "Segundo a Wikipédia: {v}",
};

// tools.norm do Python: minúsculas, sem acento, sem pontuação estranha, espaços simples
export function norm(s) {
  return s.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "")
    .replace(/[^\p{L}\p{N}_\s.+#-]/gu, " ").replace(/\s+/g, " ").trim();
}

export function parseCall(text, ferramentas) {
  const i = text.indexOf(":");
  const name = (i < 0 ? text : text.slice(0, i)).trim();
  if (!(name in ferramentas)) return null;
  return [name, i < 0 ? "" : text.slice(i + 1).trim()];
}

// Igual ao sample_next do Python: temperatura, top-k e top-p (nucleus).
export function sampleNext(logits, temperature = 0.7, topK = 40, topP = 0.9, rand = Math.random) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  if (temperature <= 0) return best;
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  const k = topK && topK < logits.length ? topK : logits.length;
  const kth = logits[order[k - 1]];
  const cand = order.filter((i) => logits[i] >= kth && logits[i] > -Infinity);
  const mx = logits[cand[0]] / temperature;
  let probs = cand.map((i) => Math.exp(logits[i] / temperature - mx));
  let sum = probs.reduce((a, b) => a + b, 0);
  probs = probs.map((p) => p / sum);
  if (topP < 1) {
    let acc = 0;
    probs = probs.map((p) => { const keep = acc <= topP; acc += p; return keep ? p : 0; });
    sum = probs.reduce((a, b) => a + b, 0);
    probs = probs.map((p) => p / sum);
  }
  let r = rand();
  for (let j = 0; j < cand.length; j++) { r -= probs[j]; if (r <= 0) return cand[j]; }
  return cand[cand.length - 1];
}

function repetitionPenalty(logits, recent, penalty) {
  if (penalty === 1 || !recent.length) return;
  for (const i of new Set(recent)) logits[i] = logits[i] > 0 ? logits[i] / penalty : logits[i] * penalty;
}

export class Lia {
  // manifest = modelo.json; buffer = pesos colados
  constructor(manifest, buffer, opts = {}) {
    this.tok = new Tokenizer(manifest.tokenizer);
    this.model = new Modelo(manifest, buffer);
    this.fixas = Object.keys(manifest.fixas || {}).length ? manifest.fixas : null;
    this.ferramentas = manifest.ferramentas;
    this.info = { passo: manifest.passo, val_loss: manifest.val_loss, copia: !!this.fixas };
    this.temperature = opts.temperature ?? 0.7;
    this.topK = opts.topK ?? 40;
    this.topP = opts.topP ?? 0.9;
    this.repetitionPenalty = opts.repetitionPenalty ?? 1.1;
    this.maxReplyTokens = opts.maxReplyTokens ?? 160;
    const t = this.tok;
    this.banned = [t.bos, t.user, t.result, t.call, t.assistant];
    this.reset();
  }

  reset() {
    this.turns = [];  // [papel, texto, ids]
    this.cache = this.model.novoCache();
    this.cacheIds = [];
    this.pending = null;
  }

  addUser(text) { this.turns.push(["user", text, this.tok.encodeTurn("user", text)]); }

  context() { return [this.tok.bos, ...this.turns.flatMap((t) => t[2])]; }

  // Garante que o cache contém `prompt` e devolve os logits do último token.
  prefill(prompt) {
    const limit = this.model.Tmax - this.maxReplyTokens;
    if (prompt.length > limit) {  // conversa longa: esquece os turnos mais antigos
      while (this.context().length + 1 > limit && this.turns.length > 1) this.turns.shift();
      prompt = [...this.context(), this.tok.assistant];
      if (prompt.length > limit) prompt = [this.tok.bos, ...prompt.slice(-(limit - 1))];
      this.cache.pos = 0; this.cacheIds = [];
    }
    let n = this.cacheIds.length;
    const same = n < prompt.length && this.cacheIds.every((id, i) => id === prompt[i]);
    if (!same) { this.cache.pos = 0; this.cacheIds = []; n = 0; }
    const logits = this.model.forward(prompt.slice(n), this.cache);
    this.cacheIds = [...prompt];
    return logits;
  }

  feed(token) {
    const logits = this.model.forward([token], this.cache);
    this.cacheIds.push(token);
    return logits;
  }

  // Gera UMA fala da Lia. Eventos: {type:"token", text} {type:"text", text} {type:"call", tool, arg, confirm}
  *step() {
    const tok = this.tok;
    let logits = this.prefill([...this.context(), tok.assistant]);
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;

    if (best === tok.call) {  // se <|call|> é o mais provável, é ferramenta
      const gen = [tok.call];
      logits = this.feed(tok.call);
      const copia = this.fixas ? new Copia(this.fixas, this.turns.map((t) => t[1]).join(" ")) : null;
      let raw = new Uint8Array(0);
      for (let s = 0; s < 48; s++) {
        for (const b of this.banned) logits[b] = -Infinity;
        let t;
        if (copia) t = copia.escolhe(logits, raw, tok);
        else { t = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[t]) t = i; }
        if (t === null || t === tok.end) break;
        gen.push(t);
        const v = tok.vocab[t], next = new Uint8Array(raw.length + v.length);
        next.set(raw); next.set(v, raw.length); raw = next;
        logits = this.feed(t);
      }
      const text = tok.decode(gen.slice(1)).trim();
      const parsed = parseCall(text, this.ferramentas);
      if (parsed) {
        this.turns.push(["call", text, [tok.assistant, ...gen, tok.end]]);
        this.pending = parsed;
        yield { type: "call", tool: parsed[0], arg: parsed[1], confirm: !!this.ferramentas[parsed[0]].confirm };
        return;
      }
      this.turns.push(["assistant", FALLBACK_TEXT, tok.encodeTurn("assistant", FALLBACK_TEXT)]);
      this.cache.pos = 0; this.cacheIds = [];
      yield { type: "token", text: FALLBACK_TEXT };
      yield { type: "text", text: FALLBACK_TEXT };
      return;
    }

    // logo depois de uma ferramenta de valor, a resposta é conferida antes de aparecer
    let must = null;
    const n = this.turns.length;
    if (n >= 2 && this.turns[n - 1][0] === "result" && this.turns[n - 2][0] === "call") {
      const name = this.turns[n - 2][1].split(":")[0].trim(), res = this.turns[n - 1][1];
      if (name in FAITHFUL && res.startsWith("ok: ") && !res.includes("nenhuma nota")) must = [name, res.slice(4)];
    }
    // Respostas recentes dela: se sair igualzinha a uma delas, gera de novo (até 2 vezes).
    // O evento "text" do fim troca o que já apareceu na tela.
    const recentes = new Set(this.turns.slice(-6).filter((t) => t[0] === "assistant").map((t) => norm(t[1])));
    const start = this.cacheIds.length, last = this.cacheIds[start - 1];
    let gen = [], text = "";
    const proibidas = [];  // 1ª palavra das tentativas que repetiram: a próxima começa diferente
    for (let tentativa = 0; tentativa < 3; tentativa++) {
      gen = [];
      const decoder = new TextDecoder("utf-8");
      const temperature = tentativa ? Math.max(this.temperature, 0.8) + 0.2 * tentativa : this.temperature;
      for (let s = 0; s < this.maxReplyTokens; s++) {
        for (const b of this.banned) logits[b] = -Infinity;
        if (!gen.length) {
          logits[tok.end] = -Infinity;  // nada de resposta vazia
          for (const b of proibidas) logits[b] = -Infinity;
        }
        repetitionPenalty(logits, gen.slice(-48), this.repetitionPenalty);
        const t = sampleNext(logits, temperature, this.topK, this.topP);
        if (t === tok.end) break;
        gen.push(t);
        const piece = decoder.decode(tok.vocab[t], { stream: true });
        if (piece && !must && !tentativa) yield { type: "token", text: piece };
        logits = this.feed(t);
      }
      const tail = decoder.decode();
      if (tail && !must && !tentativa) yield { type: "token", text: tail };
      text = tok.decode(gen).trim();
      if (must || !recentes.has(norm(text)) || tentativa === 2) break;
      // repetiu: volta o cache para o fim do prompt e sorteia de novo
      proibidas.push(gen[0]);
      this.cache.pos = start - 1;
      this.cacheIds = this.cacheIds.slice(0, start - 1);
      logits = this.feed(last);
    }
    let ids = [tok.assistant, ...gen, tok.end];
    if (must) {
      const [name, value] = must;
      if (!norm(text).includes(norm(value))) {
        text = FAITHFUL[name].replace("{v}", value);
        ids = tok.encodeTurn("assistant", text);
        this.cache.pos = 0; this.cacheIds = [];  // o cache tem a resposta errada; refaz na próxima
      }
      yield { type: "token", text };
    }
    this.turns.push(["assistant", text, ids]);
    yield { type: "text", text };
  }

  // Coloca o resultado da ferramenta pendente no histórico (quem usa executa a ferramenta).
  resolve(result) {
    this.pending = null;
    this.turns.push(["result", result, this.tok.encodeTurn("result", result)]);
  }
}
