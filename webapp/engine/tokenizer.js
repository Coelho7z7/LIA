// Tokenizador BPE em nível de bytes: o mesmo do tokenizer.py, em JavaScript.
// Tem que quebrar o texto EXATAMENTE igual ao Python (o teste de paridade confere).

// \s do Python (str.isspace) — o \s do JS não tem \x1c-\x1f nem \x85 e tem ﻿ a mais.
const WS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
// Mesmo padrão do Python: " ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+"
// \w do Python = letras + números (str.isalnum) + "_"; \d = só dígitos decimais (Nd).
export const SPLIT = new RegExp(
  ` ?[\\p{L}\\p{Nl}\\p{No}]+| ?\\p{Nd}{1,3}| ?[^${WS}\\p{L}\\p{N}_]+| ?_+|[${WS}]+(?![^${WS}])|[${WS}]+`, "gu");

export const SPECIAL_TOKENS = ["<|bos|>", "<|user|>", "<|assistant|>", "<|call|>", "<|result|>", "<|end|>"];

const utf8 = new TextEncoder();

export class Tokenizer {
  constructor({ merges, special = SPECIAL_TOKENS }) {
    this.nMerges = merges.length;
    this.ranks = new Map();  // (a, b) -> id novo; id menor = regra aprendida antes
    this.vocab = [];  // id -> Uint8Array
    for (let i = 0; i < 256; i++) this.vocab.push(Uint8Array.of(i));
    merges.forEach(([a, b], k) => {
      const id = 256 + k;
      this.ranks.set(a * 65536 + b, id);
      const va = this.vocab[a], vb = this.vocab[b], v = new Uint8Array(va.length + vb.length);
      v.set(va); v.set(vb, va.length);
      this.vocab.push(v);
    });
    this.special = {};
    special.forEach((name, k) => {
      this.special[name] = 256 + this.nMerges + k;
      this.vocab.push(utf8.encode(name));
    });
    const s = this.special;
    [this.bos, this.user, this.assistant] = [s["<|bos|>"], s["<|user|>"], s["<|assistant|>"]];
    [this.call, this.result, this.end] = [s["<|call|>"], s["<|result|>"], s["<|end|>"]];
    this.cache = new Map();
  }

  get vocabSize() { return this.vocab.length; }

  encodeChunk(bytes) {
    let ids = Array.from(bytes);
    while (ids.length >= 2) {
      let best = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < best) best = r;
      }
      if (best === Infinity) break;
      const out = [];
      for (let i = 0; i < ids.length; i++) {  // troca toda ocorrência do par, da esquerda pra direita
        if (i < ids.length - 1 && this.ranks.get(ids[i] * 65536 + ids[i + 1]) === best) { out.push(best); i++; }
        else out.push(ids[i]);
      }
      ids = out;
    }
    return ids;
  }

  // Texto comum -> ids. Nunca produz tokens especiais.
  encode(text) {
    const out = [];
    for (const m of text.matchAll(SPLIT)) {
      const chunk = m[0];
      let ids = this.cache.get(chunk);
      if (!ids) {
        ids = this.encodeChunk(utf8.encode(chunk));
        if (this.cache.size < 200000) this.cache.set(chunk, ids);
      }
      for (const i of ids) out.push(i);
    }
    return out;
  }

  // Uma fala da conversa -> ids (igual ao encode_turn do Python, sem a máscara).
  encodeTurn(role, text) {
    const body = this.encode(text);
    if (role === "user") return [this.user, ...body, this.end];
    if (role === "result") return [this.result, ...body, this.end];
    if (role === "assistant") return [this.assistant, ...body, this.end];
    if (role === "call") return [this.assistant, this.call, ...body, this.end];
    throw new Error(`papel desconhecido: ${role}`);
  }

  decodeBytes(ids) {
    let n = 0;
    for (const i of ids) n += this.vocab[i].length;
    const out = new Uint8Array(n);
    let o = 0;
    for (const i of ids) { out.set(this.vocab[i], o); o += this.vocab[i].length; }
    return out;
  }

  decode(ids) { return new TextDecoder("utf-8").decode(this.decodeBytes(ids)); }
}
