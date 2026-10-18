// Ferramentas puras da Lia (sem navegador): as mesmas do tools.py, portadas para JS.
// O backend.js usa; o teste webapp/test/ferramentas.js compara com o Python.

// ============================================================ utilidades
export const stripAccents = (s) => s.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "");
export function norm(s) { return stripAccents(s).replace(/[^\p{L}\p{N}_\s.+#-]/gu, " ").replace(/\s+/g, " ").trim(); }
export const pad = (n) => String(n).padStart(2, "0");

// ============================================================ sites e pesquisa
export const SITES = {
  "youtube": "https://www.youtube.com", "google": "https://www.google.com", "gmail": "https://mail.google.com",
  "github": "https://github.com", "netflix": "https://www.netflix.com", "wikipedia": "https://pt.wikipedia.org",
  "whatsapp web": "https://web.whatsapp.com", "whatsapp": "https://web.whatsapp.com",
  "instagram": "https://www.instagram.com", "twitter": "https://x.com", "reddit": "https://www.reddit.com",
  "linkedin": "https://www.linkedin.com", "twitch": "https://www.twitch.tv", "amazon": "https://www.amazon.com.br",
  "mercado livre": "https://www.mercadolivre.com.br", "g1": "https://g1.globo.com", "globo": "https://www.globo.com",
  "google drive": "https://drive.google.com", "drive": "https://drive.google.com",
  "google maps": "https://maps.google.com", "maps": "https://maps.google.com",
  "tradutor": "https://translate.google.com", "google tradutor": "https://translate.google.com",
  "canva": "https://www.canva.com", "facebook": "https://www.facebook.com", "tiktok": "https://www.tiktok.com",
  "pinterest": "https://www.pinterest.com", "prime video": "https://www.primevideo.com",
  "disney plus": "https://www.disneyplus.com", "disney+": "https://www.disneyplus.com", "max": "https://www.max.com",
  "stack overflow": "https://stackoverflow.com", "duolingo": "https://www.duolingo.com",
  "claude": "https://claude.ai", "google agenda": "https://calendar.google.com",
  "agenda": "https://calendar.google.com", "shopee": "https://shopee.com.br", "ifood": "https://www.ifood.com.br",
  "spotify web": "https://open.spotify.com", "nubank": "https://nubank.com.br", "uol": "https://www.uol.com.br",
};
export const DOMAIN = /^(https?:\/\/)?([\p{L}\p{N}_-]+\.)+[a-z]{2,}(\/\S*)?$/u;

// difflib.SequenceMatcher.ratio (sem "junk"): 2*M/T com os maiores blocos em comum
export function ratio(a, b) {
  const match = (alo, ahi, blo, bhi) => {
    let best = 0, bi = alo, bj = blo;
    for (let i = alo; i < ahi; i++) {
      for (let j = blo; j < bhi; j++) {
        let k = 0;
        while (i + k < ahi && j + k < bhi && a[i + k] === b[j + k]) k++;
        if (k > best) { best = k; bi = i; bj = j; }
      }
    }
    if (!best) return 0;
    return best + match(alo, bi, blo, bj) + match(bi + best, ahi, bj + best, bhi);
  };
  return a.length + b.length ? (2 * match(0, a.length, 0, b.length)) / (a.length + b.length) : 1;
}

export function siteUrl(q) {
  q = norm(q).replace(/^(o |a )?(site |pagina )?(do |da |de )?/, "");
  if (q in SITES) return SITES[q];
  if (DOMAIN.test(q)) return q.startsWith("http") ? q : "https://" + q;
  let best = null, bs = 0.78;
  for (const k of Object.keys(SITES)) { const r = ratio(q, k); if (r >= bs) { bs = r; best = k; } }
  return best ? SITES[best] : null;
}

// ============================================================ hora, data, contas
export const DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"];
export const MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
  "outubro", "novembro", "dezembro"];
export const formatarData = (d) => `${DIAS[(d.getDay() + 6) % 7]}, ${d.getDate()} de ${MESES[d.getMonth()]} de ${d.getFullYear()}`;

export function toExpr(expr) {  // _to_python do tools.py
  let s = stripAccents(expr.replaceAll("×", "*").replaceAll("÷", "/").replaceAll("^", "**").replaceAll("√", " raiz de "));
  s = s.replaceAll("?", " ").replaceAll("=", " ").replace(/\s+/g, " ").trim();
  s = s.replace(/^(quanto e|quanto da|calcula|calcule|resolve)\s+/, "");
  s = s.replace(/(\d),(\d)/g, "$1.$2");
  s = s.replace(/(\d+(?:\.\d+)?)\s*(%|por cento)\s*(de|do|da|sobre)\s*/g, "$1/100*");
  s = s.replace(/(\d+(?:\.\d+)?)\s*(%|por cento)/g, "($1/100)");
  for (const [a, b] of [["raiz quadrada de", "sqrt "], ["raiz de", "sqrt "], ["raiz", "sqrt "],
    ["ao quadrado", "**2"], ["ao cubo", "**3"], ["elevado ao", "**"], ["elevado a", "**"],
    ["vezes", "*"], ["multiplicado por", "*"], ["dividido por", "/"], ["divido por", "/"],
    ["sobre", "/"], ["mais", "+"], ["menos", "-"], ["seno de", "sin "], ["cosseno de", "cos "], ["log de", "log "]]) {
    s = s.replaceAll(a, b);
  }
  s = s.replace(/(?<=[\d)\s])x(?=[\s\d(])/g, "*");
  s = s.replace(/\b(sqrt|sin|cos|tan|log)\s+(\d+(?:\.\d+)?)/g, "$1($2)");
  return s.trim();
}

export const roundHalfEven = (v) => { const f = Math.floor(v), d = v - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };
export const FUNCS = { sqrt: Math.sqrt, sin: Math.sin, cos: Math.cos, tan: Math.tan, log: Math.log10, ln: Math.log, abs: Math.abs,
  round: (v, n) => (n === undefined ? roundHalfEven(v) : roundHalfEven(v * 10 ** n) / 10 ** n) };
export const CONSTS = { pi: Math.PI, e: Math.E };

// Só aritmética, como o _eval do Python (precedência do Python: -2**2 = -4, ** à direita)
export function avaliar(src) {
  const toks = src.match(/\d+(?:\.\d*)?(?:e[+-]?\d+)?|\.\d+(?:e[+-]?\d+)?|[a-z_]\w*|\*\*|\/\/|[-+*/%(),]|\S/gi) || [];
  let i = 0;
  const peek = () => toks[i], next = () => toks[i++];
  const fail = () => { throw new Error("expressão não suportada"); };
  const pyMod = (a, b) => { if (b === 0) fail(); return a - b * Math.floor(a / b); };
  function expr() {
    let v = term();
    while (peek() === "+" || peek() === "-") v = next() === "+" ? v + term() : v - term();
    return v;
  }
  function term() {
    let v = unary();
    for (;;) {
      const op = peek();
      if (op === "*") { next(); v *= unary(); }
      else if (op === "/") { next(); const b = unary(); if (b === 0) fail(); v /= b; }
      else if (op === "//") { next(); const b = unary(); if (b === 0) fail(); v = Math.floor(v / b); }
      else if (op === "%") { next(); v = pyMod(v, unary()); }
      else return v;
    }
  }
  function unary() {
    if (peek() === "-") { next(); return -unary(); }
    if (peek() === "+") { next(); return unary(); }
    return power();
  }
  function power() {
    const a = atom();
    if (peek() !== "**") return a;
    next();
    const b = unary();
    if (Math.abs(b) > 100 || Math.abs(a) > 1e6) throw new Error("número grande demais");
    return a ** b;
  }
  function atom() {
    const t = next();
    if (t === undefined) fail();
    if (t === "(") { const v = expr(); if (next() !== ")") fail(); return v; }
    if (/^[\d.]/.test(t)) { const v = Number(t); if (Number.isNaN(v)) fail(); return v; }
    if (/^[a-z_]/i.test(t)) {
      if (peek() === "(" && t in FUNCS) {
        next();
        const args = [];
        if (peek() !== ")") { args.push(expr()); while (peek() === ",") { next(); args.push(expr()); } }
        if (next() !== ")") fail();
        return FUNCS[t](...args);
      }
      if (t in CONSTS) return CONSTS[t];
    }
    return fail();
  }
  const v = expr();
  if (i !== toks.length) fail();
  return v;
}

export function formatG(v) {  // f"{v:.6g}" do Python, com vírgula decimal
  if (v === 0) return "0";
  const [m, e] = v.toExponential(5).split("e");
  const exp = Number(e);
  let s;
  if (exp < -4 || exp >= 6) {
    const mant = m.replace(/\.?0+$/, "");
    s = `${mant}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  } else {
    s = v.toFixed(Math.max(0, 5 - exp)).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  }
  return s.replace(".", ",");
}

// Conta só com inteiros e + - * ** // %: inteiro exato (BigInt), igual ao int do Python.
// Devolve null quando a conta não é desse tipo (aí vale a de ponto flutuante).
export function avaliarInteiro(src) {
  const toks = src.match(/\d+|\*\*|\/\/|[-+*%()]|\S/g) || [];
  if (toks.some((t) => !/^(\d+|\*\*|\/\/|[-+*%()])$/.test(t))) return null;
  let i = 0;
  const peek = () => toks[i], next = () => toks[i++];
  const fail = () => { throw new Error("expressão não suportada"); };
  const abs = (a) => (a < 0n ? -a : a);
  function expr() {
    let v = term();
    while (peek() === "+" || peek() === "-") v = next() === "+" ? v + term() : v - term();
    return v;
  }
  function term() {
    let v = unary();
    for (;;) {
      const op = peek();
      if (op === "*") { next(); v *= unary(); }
      else if (op === "//" || op === "%") {
        next();
        const b = unary();
        if (b === 0n) fail();
        const q = v / b, r = v % b, ajusta = r !== 0n && (r < 0n) !== (b < 0n);  // Python arredonda para baixo
        v = op === "//" ? (ajusta ? q - 1n : q) : (ajusta ? r + b : r);
      } else return v;
    }
  }
  function unary() {
    if (peek() === "-") { next(); return -unary(); }
    if (peek() === "+") { next(); return unary(); }
    return power();
  }
  function power() {
    const a = atom();
    if (peek() !== "**") return a;
    next();
    const b = unary();
    if (abs(b) > 100n || abs(a) > 1000000n) throw new Error("número grande demais");
    if (b < 0n) throw new RangeError("expoente negativo");  // vira fração: fica com o ponto flutuante
    return a ** b;
  }
  function atom() {
    const t = next();
    if (t === "(") { const v = expr(); if (next() !== ")") fail(); return v; }
    if (t !== undefined && /^\d+$/.test(t)) return BigInt(t);
    return fail();
  }
  try {
    const v = expr();
    return i === toks.length ? v : null;
  } catch (e) {
    if (e instanceof RangeError) return null;
    throw e;
  }
}

export function calcularValor(expr) {
  const src = toExpr(expr), big = avaliarInteiro(src);
  if (big !== null) {
    const digits = (big < 0n ? -big : big).toString();
    const grouped = digits.length > 4 ? digits.replace(/\B(?=(\d{3})+(?!\d))/g, ".") : digits;
    return (big < 0n ? "-" : "") + grouped;
  }
  const v = avaliar(src);
  if (!Number.isFinite(v)) throw new Error("resultado inválido");
  if (Number.isInteger(v) && Math.abs(v) < 1e15) {
    return Math.abs(v) >= 10000 ? v.toLocaleString("en-US").replaceAll(",", ".") : String(v);
  }
  return formatG(v);
}

// ============================================================ lembretes
export const NUM_WORDS = { um: 1, uma: 1, dois: 2, duas: 2, tres: 3, quatro: 4, cinco: 5, seis: 6, sete: 7, oito: 8, nove: 9,
  dez: 10, quinze: 15, vinte: 20, trinta: 30, quarenta: 40, cinquenta: 50, sessenta: 60 };

// 'às 15h', 'às 7:30', 'amanhã', 'amanhã 9h', 'hoje às 20h', 'meio-dia' -> [amanhã?, hora, minuto] (null se não marca horário)
function horario(s) {
  s = stripAccents(s);
  const amanha = /\bamanha\b/.test(s), hoje = /\bhoje\b/.test(s);
  let h = null, mi = null;
  if (/\bmeio[ -]?dia\b/.test(s)) { h = 12; mi = 0; }
  else if (/\bmeia[ -]?noite\b/.test(s)) { h = 0; mi = 0; }
  else {
    const m = s.match(/\b(\d{1,2}):(\d{2})\b/) || s.match(/\b(?:as|ate as)\s+(\d{1,2})(?:\s*(?:h|horas?)\s*(\d{2})?)?\b/)
      || ((amanha || hoje) && s.match(/\b(\d{1,2})\s*h(?:oras?)?\s*(\d{2})?\b/));
    if (m) {
      h = Number(m[1]); mi = Number(m[2] || 0);
      if (h < 12 && /\bda (tarde|noite)\b/.test(s)) h += 12;  // "às 3 da tarde"
    }
  }
  if (h === null) return amanha ? [true, 9, 0] : null;  // "amanhã" sem hora: às 9h
  if (!(h >= 0 && h <= 23 && mi >= 0 && mi <= 59)) return null;
  return [amanha, h, mi];
}

export function tempoAbsoluto(s, agora = new Date()) {
  const hr = horario(s);
  if (!hr) return null;
  const [amanha, h, mi] = hr;
  const alvo = new Date(agora.getFullYear(), agora.getMonth(), agora.getDate(), h, mi, 0, 0);
  if (amanha || alvo <= agora) alvo.setDate(alvo.getDate() + 1);
  return Math.trunc((alvo - agora) / 1000);
}

// O modelo às vezes perde o "às 15h"/"amanhã" ao copiar: se a fala marca horário e o argumento não, vale o da fala.
export function ancorarLembrete(arg, fala) {
  const hr = horario(fala);
  if (!hr) return arg;
  const [amanha, h, mi] = hr;
  const expr = [...(amanha ? ["amanhã"] : []), `às ${h}h` + (mi ? String(mi).padStart(2, "0") : "")].join(" ");
  const i = arg.indexOf("|");
  if (i >= 0) {
    const atual = horario(arg.slice(0, i));
    return atual && atual.join() === hr.join() ? arg : `${expr} | ${arg.slice(i + 1).trim()}`;
  }
  try { parseDuracao(arg); return expr; } catch { return `${expr} | ${arg.trim()}`; }
}

export function parseDuracao(s, agora = new Date()) {
  const t = tempoAbsoluto(s, agora);
  if (t !== null) {
    if (t <= 0 || t > 7 * 86400) throw new Error("horário fora do limite");
    return t;
  }
  if (/\bas\s+\d/.test(stripAccents(s))) throw new Error("horário inválido");
  s = norm(s);
  if (s.includes("meia hora")) return 1800;
  s = s.replace(new RegExp(`\\b(${Object.keys(NUM_WORDS).join("|")})\\b`, "g"), (w) => String(NUM_WORDS[w]));
  const hm = s.match(/(\d+)\s*h(?:oras?)?\s*e?\s*(\d+)/);
  if (hm) return Number(hm[1]) * 3600 + Number(hm[2]) * 60;
  let total = 0, found = false;
  for (const m of s.matchAll(/(\d+(?:[.,]\d+)?)\s*(horas?|h|minutos?|mins?|m|segundos?|seg|s)\b/g)) {
    const n = Number(m[1].replace(",", "."));
    total += n * (m[2].startsWith("h") ? 3600 : m[2].startsWith("s") ? 1 : 60);
    found = true;
  }
  if (!found) {
    const m = s.match(/\d+(?:[.,]\d+)?/);
    if (!m) throw new Error("sem número");
    total = Number(m[0].replace(",", ".")) * 60;  // número solto = minutos
  }
  if (total <= 0 || total > 7 * 86400) throw new Error("duração fora do limite");
  return Math.trunc(total);
}

export function formatarDuracao(sec) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60, parts = [];
  if (h) parts.push(`${h} hora` + (h > 1 ? "s" : ""));
  if (m) parts.push(`${m} minuto` + (m > 1 ? "s" : ""));
  if (s && !h) parts.push(`${s} segundo` + (s > 1 ? "s" : ""));
  return parts.join(" e ");
}


// recordar: os fatos mais parecidos com a pergunta (até 2)
const STOP = new Set("o a os as um uma de do da dos das e eu meu minha me que qual quais é eh sou em no na pra para com voce vc sobre mim lembra sabe".split(" "));
export function recordarEm(fs, arg) {
  if (!fs.length) return "erro: não tenho nada sobre isso";
  const words = new Set(norm(arg).split(" ").filter((w) => w && !STOP.has(w)));
  if (!words.size) return "ok: " + fs.slice(-5).join("; ");
  const score = (f) => {
    const fw = new Set(norm(f).split(" "));
    let n = 0;
    for (const w of words) if (fw.has(w) || (w.length >= 5 && [...fw].some((x) => x.startsWith(w.slice(0, 5))))) n++;
    return n;
  };
  const hits = fs.map((f) => [f, score(f)]).sort((a, b) => b[1] - a[1]).filter(([, sc]) => sc > 0).slice(0, 2).map(([f]) => f);
  return hits.length ? "ok: " + hits.join("; ") : "erro: não tenho nada sobre isso";
}
// "meu nome é X" e "meu nome é Y" têm a mesma chave: o novo substitui o antigo
export const chave = (f) => norm(f).split(" ").slice(0, 3).join(" ");

// consultar (Wikipédia): o resumo_wiki/assunto_wiki do tools.py. Mexeu num, mexe no outro.
const ABREV = new Set(["dr", "dra", "sr", "sra", "srta", "st", "jr", "prof", "profa", "a.c", "d.c", "séc", "vol", "cap", "n.º", "nº"]);
const ncp = (s) => [...s].length;  // tamanho em caracteres, como o len() do Python (emoji conta 1)
export function resumoWiki(texto, limite = 260) {
  let t = texto.trim().split("\n")[0].replace(/\s*\[[^[\]]*\]/g, "");  // pronúncia: [kɐ̃ˈbɛʁɐ]
  let antes = null;
  while (antes !== t) {  // parênteses sem número (variante, nome em inglês) saem; com data ficam
    antes = t;
    t = t.replace(/\s*\(([^()]*)\)/g, (m, d) => (/[0-9]/.test(d) ? m : ""));
  }
  t = t.replace(/\s+/g, " ").replace(/\s+([,.;:])/g, "$1").trim();
  const frases = [];
  let ini = 0;
  for (const m of t.matchAll(/[.!?](?=\s+[A-ZÀ-Ý0-9"“(])/g)) {
    const w = t.slice(ini, m.index).split(" ").pop();
    if (ABREV.has(w.toLowerCase()) || (ncp(w) === 1 && /\p{L}/u.test(w))) continue;  // "Dr.", "a.C.", "J. R."
    frases.push(t.slice(ini, m.index + 1).trim());
    ini = m.index + 1;
  }
  if (t.slice(ini).trim()) frases.push(t.slice(ini).trim());
  let out = frases[0] || "";
  if (frases.length > 1 && ncp(out) < 80 && ncp(out) + 1 + ncp(frases[1]) <= limite) out += " " + frases[1];
  if (ncp(out) > limite) {
    let corte = [...out].slice(0, limite).join("");
    const k = corte.lastIndexOf(", ");
    corte = k >= 0 && ncp(corte.slice(0, k)) > limite * 0.6 ? corte.slice(0, k) : corte.slice(0, corte.lastIndexOf(" "));
    out = corte.replace(/[ ,;:]+$/, "") + "…";
  }
  return out;
}
export const assuntoWiki = (arg) => arg.replace(/^[ ?!.\t]+|[ ?!.\t]+$/g, "").replace(/^(o|a|os|as|um|uma)\s+/i, "").trim();
