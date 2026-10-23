// Cópia forçada na chamada de ferramenta: o mesmo do copia.py.
// A cada token da chamada, fica com o mais provável que mantém o argumento válido:
// palavras só da conversa (no máximo uma vez a mais do que foram ditas) ou do vocabulário
// fixo da ferramenta, e pontuação só a que já apareceu na conversa.

const W = /[\p{L}\p{N}_]+/gu;  // \w do Python
const ENDS_WORD = /[\p{L}\p{N}_]$/u;
const utf8 = new TextEncoder();
const strict = new TextDecoder("utf-8", { fatal: true });

export function palavras(s) { return s.toLowerCase().match(W) || []; }

function startsWith(word, prefix) {  // os dois em bytes
  if (prefix.length > word.length) return false;
  for (let i = 0; i < prefix.length; i++) if (word[i] !== prefix[i]) return false;
  return true;
}

// Bytes no fim que ainda podem virar um caractere (acento pela metade) — igual ao decodificador do Python.
function incompleteTail(raw) {
  for (let back = 1; back <= Math.min(3, raw.length); back++) {
    const b = raw[raw.length - back];
    if ((b & 0xc0) === 0x80) continue;  // byte de continuação: o início está mais atrás
    const need = b >= 0xc2 && b <= 0xdf ? 2 : b >= 0xe0 && b <= 0xef ? 3 : b >= 0xf0 && b <= 0xf4 ? 4 : 0;
    if (!need || back >= need) return 0;
    if (back >= 2) {  // segundo byte tem faixa própria para alguns iniciais
      const s = raw[raw.length - back + 1];
      const lo = b === 0xe0 ? 0xa0 : b === 0xf0 ? 0x90 : 0x80, hi = b === 0xed ? 0x9f : b === 0xf4 ? 0x8f : 0xbf;
      if (s < lo || s > hi) return 0;
    }
    return back;
  }
  return 0;
}

function decodeStrict(raw, final) {  // {text, tail} ou null se a sequência é inválida
  const n = final ? 0 : incompleteTail(raw);
  try {
    return { text: strict.decode(raw.subarray(0, raw.length - n)), tail: raw.subarray(raw.length - n) };
  } catch {
    return null;
  }
}

export class Copia {
  constructor(fixas, contexto) {
    this.fixas = {};
    for (const [t, ws] of Object.entries(fixas)) this.fixas[t] = new Map(ws.map((w) => [w, utf8.encode(w)]));
    const n = new Map();
    for (const w of palavras(contexto)) n.set(w, (n.get(w) || 0) + 1);
    this.saldo = new Map([...n].map(([w, k]) => [w, { k: k + 1, b: utf8.encode(w) }]));
    this.simbolos = new Set([...contexto.toLowerCase().replace(W, ""), " ", ",", "|"]);
  }

  // raw: bytes da chamada até aqui ("abrir_app: Pow..."); final: pode terminar aqui?
  aceita(raw, final = false) {
    const d = decodeStrict(raw, final);
    if (!d) return false;
    const { text, tail } = d;
    const c = text.indexOf(":");
    if (c < 0) return !final;  // ainda no nome da ferramenta
    const fixed = this.fixas[text.slice(0, c).trim()] || new Map();
    const arg = text.slice(c + 1).toLowerCase();
    for (const ch of arg.replace(W, "")) if (!this.simbolos.has(ch)) return false;
    const ws = arg.match(W) || [];
    if (!ws.length && !tail.length) return true;
    const endsWord = ENDS_WORD.test(arg);
    let done, last;
    if (tail.length) {  // a última palavra termina num acento pela metade
      if (endsWord) {
        const lb = utf8.encode(ws[ws.length - 1]);
        last = new Uint8Array(lb.length + tail.length); last.set(lb); last.set(tail, lb.length);
        done = ws.slice(0, -1);
      } else { last = tail; done = ws; }
    } else if (endsWord && !final) {
      done = ws.slice(0, -1); last = utf8.encode(ws[ws.length - 1]);
    } else { done = ws; last = null; }
    // Palavra fixa da ferramenta só vale no começo ("meu nome é", "eu trabalho como"); depois que a cópia da
    // conversa começou, toda palavra tem que vir da conversa ("analista de moro" não passa).
    const used = new Map();
    let copiando = false;
    for (const w of done) {
      if (fixed.has(w) && !copiando) continue;
      used.set(w, (used.get(w) || 0) + 1);
      copiando = copiando || !fixed.has(w);
      if (used.get(w) > (this.saldo.get(w)?.k || 0)) return false;
    }
    if (last === null) return true;
    // a palavra aberta tem que poder virar uma da conversa que ainda tem saldo (ou fixa, se ainda no começo)
    if (!copiando) for (const b of fixed.values()) if (startsWith(b, last)) return true;
    for (const [w, s] of this.saldo) if (s.k > (used.get(w) || 0) && startsWith(s.b, last)) return true;
    return false;
  }

  // O token mais provável que mantém a chamada válida (null se nenhum serve).
  escolhe(logits, gen, tok) {
    const nTok = 256 + tok.nMerges;
    const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
    for (const c of order) {
      let ok;
      if (c === tok.end) ok = this.aceita(gen, true);
      else if (c >= nTok) continue;  // especiais (fora o fim) nunca entram numa chamada
      else {
        const v = tok.vocab[c], next = new Uint8Array(gen.length + v.length);
        next.set(gen); next.set(v, gen.length);
        ok = this.aceita(next);
      }
      if (ok) return c;
    }
    return null;
  }
}
