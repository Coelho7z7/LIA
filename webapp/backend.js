// O "web.py" da versão web: responde aos mesmos endereços /api/... que a interface usa,
// só que dentro da própria página. O modelo roda no worker (engine/worker.js); as ferramentas
// rodam aqui (são as do tools.py, portadas). Memória, notas e lembretes ficam neste navegador.

import { ancorarLembrete, assuntoWiki, calcularValor, chave, formatarData, formatarDuracao, norm, pad, parseDuracao, recordarEm,
  resumoWiki, siteUrl } from "./ferramentas.js";

const BASE = new URL(".", import.meta.url);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* sem espaço */ } },
};

// ============================================================ modelo (worker)
const worker = new Worker(new URL("engine/worker.js", BASE), { type: "module" });
const handlers = new Map();
let seq = 0;
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "progresso") return carga(m.feito, m.total);
  handlers.get(m.id)?.(m);
};

// Um pedido ao worker: async iterator dos eventos; o último é {type:"fim"}.
function pedir(type, data = {}) {
  const id = ++seq, fila = [];
  let acorda = null;
  handlers.set(id, (m) => { fila.push(m); if (acorda) { acorda(); acorda = null; } });
  worker.postMessage({ id, type, ...data });
  return (async function* () {
    for (;;) {
      while (fila.length) {
        const m = fila.shift();
        const last = m.type === "fim" || m.type === "pronta" || m.type === "erro";
        if (last) handlers.delete(id);  // antes do yield: quem consome pode sair do laço no "fim"
        if (m.type === "erro") throw new Error(m.message);
        yield m;
        if (last) return;
      }
      await new Promise((r) => { acorda = r; });
    }
  })();
}
async function ultimo(it) { let last = null; for await (const m of it) last = m; return last; }

let manifest = null;
const manifestPronto = fetch(new URL("modelo/modelo.json", BASE), { cache: "no-cache" }).then((r) => r.json())
  .then((m) => { manifest = m; return m; });
const pronta = ultimo(pedir("carregar", { base: new URL("modelo", BASE).href, opts: { temperature: store.get("lia-temp", 0.7) } }))
  .then((m) => { carga(1, 1, true); return m.info; })
  .catch((e) => { carga(0, 0, false, e.message); throw e; });
// Mudanças no histórico (reabrir conversa, nova conversa) entram numa fila que roda quando o
// modelo fica pronto: a página não espera o download para abrir.
let fila = pronta.catch(() => {});
function emFila(op) { const p = fila.then(op); fila = p.catch(() => {}); return p; }

// Barra de carregamento (só aparece enquanto baixa / monta o modelo)
let bar = null;
function carga(feito, total, pronto = false, erro = null) {
  if (!bar) {
    bar = document.createElement("div");
    bar.style.cssText = "position:fixed;left:50%;top:12px;transform:translateX(-50%);z-index:60;max-width:calc(100% - 32px);" +
      "padding:8px 14px;border-radius:12px;background:var(--panel,#1b1b1e);color:var(--ink,#eee);border:1px solid var(--line,#333);" +
      "font:13px/1.4 system-ui,sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.18);transition:opacity .4s";
    document.body.appendChild(bar);
  }
  if (erro) { bar.textContent = "Não consegui carregar a Lia: " + erro; return; }
  if (pronto) { bar.style.opacity = "0"; setTimeout(() => bar.remove(), 500); return; }
  const mb = (n) => (n / 2 ** 20).toFixed(0);
  bar.textContent = feito >= total ? "Preparando a Lia…" : `Baixando a Lia: ${mb(feito)} de ${mb(total)} MB (só na primeira vez)`;
}

// Abre numa aba nova. Se o navegador bloquear, o cartão mostra o link para clicar.
function abrir(url) {
  let w = null;
  try { w = window.open(url, "_blank"); if (w) w.opener = null; } catch { w = null; }
  return url;
}

// ============================================================ lembretes
const EVENTS = [];  // lembretes disparados, entregues à interface na próxima consulta
const ATIVOS = new Map();  // id -> {texto, quando, timer}
let lastReminderId = null;

function salvarLembretes() { store.set("lia-lembretes", [...ATIVOS].map(([id, r]) => ({ id, texto: r.texto, quando: r.quando }))); }
function agendar(id, texto, quando) {
  const timer = setTimeout(() => { ATIVOS.delete(id); salvarLembretes(); EVENTS.push({ type: "reminder", text: texto }); },
    Math.max(0, quando - Date.now()));
  ATIVOS.set(id, { texto, quando, timer });
}
for (const r of store.get("lia-lembretes", [])) agendar(r.id, r.texto, r.quando);  // reabriu a página: arma de novo

function cancelarLembrete(id) {
  const r = ATIVOS.get(id);
  if (!r) return false;
  clearTimeout(r.timer); ATIVOS.delete(id); salvarLembretes();
  return true;
}
const lembretesAtivos = () => [...ATIVOS].map(([id, r]) => ({ id, texto: r.texto, faltam: Math.max(0, Math.floor((r.quando - Date.now()) / 1000)) }))
  .sort((a, b) => a.faltam - b.faltam);

// ============================================================ memória e notas (neste navegador)
const fatos = () => store.get("lia-fatos", []);
const notas = () => store.get("lia-notas", []);
const REPLACED = new Map();  // fato novo -> fatos antigos que ele substituiu (para o "desfazer")
const memoria = () => ({ fatos: fatos(), notas: notas().map((n) => n.t) });

function apagarMemoria(tipo, i) {
  const key = tipo === "fatos" ? "lia-fatos" : tipo === "notas" ? "lia-notas" : null;
  if (!key) return;
  const lista = store.get(key, []);
  store.set(key, i === null || i === undefined ? [] : lista.filter((_, k) => k !== i));
}

function desfazer(tipo, ref) {
  if (tipo === "lembrete") return cancelarLembrete(ref) ? "lembrete cancelado" : "esse lembrete já tinha disparado";
  if (tipo === "fato") {
    const fs = fatos();
    if (!fs.includes(ref)) return "isso já não estava guardado";
    store.set("lia-fatos", [...fs.filter((f) => f !== ref), ...(REPLACED.get(ref) || [])]);
    REPLACED.delete(ref);
    return "esqueci isso";
  }
  if (tipo === "nota") {
    const ns = notas();
    for (let k = ns.length - 1; k >= 0; k--) {
      if (ns[k].t === ref) { ns.splice(k, 1); store.set("lia-notas", ns); return "nota apagada"; }
    }
    return "nota não encontrada";
  }
  return "não sei desfazer isso";
}

// ============================================================ clima (Open-Meteo)
const WMO = { 0: "céu limpo", 1: "quase limpo", 2: "parcialmente nublado", 3: "nublado", 45: "neblina", 48: "neblina",
  51: "garoa fraca", 53: "garoa", 55: "garoa forte", 61: "chuva fraca", 63: "chuva", 65: "chuva forte",
  66: "chuva congelante", 67: "chuva congelante", 71: "neve fraca", 73: "neve", 75: "neve forte",
  80: "pancadas de chuva", 81: "pancadas de chuva", 82: "pancadas fortes", 95: "tempestade",
  96: "tempestade com granizo", 99: "tempestade com granizo" };

async function clima(arg) {
  let cidade = arg.replace(/^[\s?]+|[\s?]+$/g, "").replace(/^(em |no |na |de |pra |para )/i, "");
  if (!cidade) cidade = store.get("lia-config", {}).cidade || "";
  if (!cidade) return "erro: não sei sua cidade";
  let g, w;
  try {
    const geo = await (await fetch("https://geocoding-api.open-meteo.com/v1/search?count=1&language=pt&name=" + encodeURIComponent(cidade))).json();
    if (!geo.results || !geo.results.length) return `erro: não achei a cidade ${cidade}`;
    g = geo.results[0];
    w = await (await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${g.latitude}&longitude=${g.longitude}` +
      "&current=temperature_2m,weather_code&daily=temperature_2m_max,temperature_2m_min," +
      "precipitation_probability_max&timezone=auto&forecast_days=1")).json();
  } catch {
    return "erro: sem internet";
  }
  const cur = w.current, day = w.daily, r0 = (v) => Math.round(v).toFixed(0);
  let texto = `${g.name}: ${r0(cur.temperature_2m)}°C, ${WMO[cur.weather_code] || "tempo instável"}, ` +
    `máx ${r0(day.temperature_2m_max[0])}°C, mín ${r0(day.temperature_2m_min[0])}°C`;
  const chuva = (day.precipitation_probability_max || [null])[0];
  if (chuva !== null && chuva !== undefined) texto += `, chuva ${chuva}%`;
  return "ok: " + texto;
}

// ============================================================ Wikipédia (o consultar do tools.py)
async function consultar(arg) {
  const q = assuntoWiki(arg);
  if (!q) return { text: "erro: consultar o quê?" };
  const p = new URLSearchParams({ action: "query", format: "json", formatversion: "2", generator: "search", gsrsearch: q,
    gsrlimit: "3", prop: "extracts|pageprops", exintro: "1", explaintext: "1", exlimit: "3", ppprop: "disambiguation",
    redirects: "1", origin: "*" });
  let d;
  try {
    // a Wikimedia pede identificação; o navegador não deixa trocar o User-Agent, então vai no Api-User-Agent
    d = await (await fetch("https://pt.wikipedia.org/w/api.php?" + p, { headers: { "Api-User-Agent": "Lia/1.0 (https://lia-ai.pages.dev)" } })).json();
  } catch {
    return { text: "erro: sem internet" };
  }
  const pages = ((d.query && d.query.pages) || []).filter((pg) => !pg.pageprops && (pg.extract || "").trim())
    .sort((a, b) => (a.index ?? 99) - (b.index ?? 99));  // pula desambiguação
  if (!pages.length) return { text: `erro: não achei ${q} na Wikipédia` };
  const alvo = pages.find((pg) => norm(pg.title) === norm(q)) || pages[0];  // título exato ganha
  const resumo = resumoWiki(alvo.extract);
  if (!resumo) return { text: `erro: não achei ${q} na Wikipédia` };
  return { text: "ok: " + resumo, titulo: alvo.title,
    link: "https://pt.wikipedia.org/wiki/" + encodeURIComponent(alvo.title.replace(/ /g, "_")) };
}

// ============================================================ ferramentas
// As que mexem no PC (apps, pastas, volume, mídia, bloquear) só funcionam com o Lia Conector.
const SO_PC = new Set(["abrir_app", "abrir_pasta", "volume", "midia", "bloquear"]);
const FILLER = /^(o|a|os|as|um|uma|meu|minha|app|aplicativo|programa|jogo|do|da|de|no|na)\s+/;
function semEnchimento(arg) { let q = norm(arg); while (FILLER.test(q)) q = q.replace(FILLER, ""); return q; }
const TEXTO_SO_PC = "Isso eu só consigo fazer no seu PC, com o Lia Conector instalado (ele tá chegando). " +
  "Aqui no site eu pesquiso, consulto a Wikipédia, abro sites, faço contas, lembretes, notas e vejo o clima.";

// Devolve {text, link?}. link: endereço que o cartão mostra para clicar (se a aba nova for bloqueada).
async function rodarFerramenta(name, arg) {
  arg = (arg || "").trim();
  try {
    switch (name) {
      case "abrir_site": {
        const url = siteUrl(arg);
        if (!url) return { text: `erro: não conheço o site ${arg}` };
        return { text: `ok: abri ${new URL(url).host.replace(/^www\./, "")}`, link: abrir(url) };
      }
      case "pesquisar":
        if (!arg) return { text: "erro: pesquisar o quê?" };
        return { text: "ok: pesquisa aberta no navegador", link: abrir("https://www.google.com/search?q=" + encodeURIComponent(arg)) };
      case "youtube":
        if (!arg) return rodarFerramenta("abrir_site", "youtube");
        return { text: "ok: busca aberta no youtube", link: abrir("https://www.youtube.com/results?search_query=" + encodeURIComponent(arg)) };
      case "hora": { const d = new Date(); return { text: `ok: ${pad(d.getHours())}:${pad(d.getMinutes())}` }; }
      case "data": return { text: "ok: " + formatarData(new Date()) };
      case "calcular":
        try { return { text: "ok: " + calcularValor(arg) }; } catch { return { text: "erro: não consegui calcular isso" }; }
      case "lembrete": {
        const i = arg.indexOf("|");
        const tempo = i < 0 ? arg : arg.slice(0, i), texto = (i < 0 ? "" : arg.slice(i + 1)).trim() || "o timer acabou";
        let sec;
        try { sec = parseDuracao(tempo); } catch { return { text: "erro: não entendi o tempo" }; }
        const id = Math.random().toString(16).slice(2, 10);
        agendar(id, texto, Date.now() + sec * 1000);
        salvarLembretes();
        lastReminderId = id;
        return { text: `ok: em ${formatarDuracao(sec)}: ${texto}` };
      }
      case "anotar": {
        if (!arg) return { text: "erro: anotar o quê?" };
        const d = new Date(), q = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
        store.set("lia-notas", [...notas(), { q, t: arg }]);
        return { text: "ok: anotado" };
      }
      case "ver_notas": {
        const ns = notas().map((n) => n.t);
        return { text: ns.length ? "ok: " + ns.slice(-5).join("; ") : "ok: nenhuma nota ainda" };
      }
      case "lembrar": {
        if (!arg) return { text: "erro: lembrar o quê?" };
        const fs = fatos(), k = chave(arg);  // fato novo sobre o mesmo assunto substitui o antigo
        REPLACED.set(arg, fs.filter((f) => chave(f) === k && f !== arg));
        store.set("lia-fatos", [...fs.filter((f) => chave(f) !== k), arg]);
        const m = arg.match(/moro em ([\p{L}\p{N}_\s'-]+)/iu);
        if (m) store.set("lia-config", { ...store.get("lia-config", {}), cidade: m[1].trim() });
        return { text: "ok: guardei" };
      }
      case "recordar": return { text: recordarEm(fatos(), arg) };
      case "clima": return { text: await clima(arg) };
      case "consultar": return await consultar(arg);
      default: return { text: `erro: ferramenta ${name} não existe` };
    }
  } catch (e) {
    return { text: `erro: ${e.message}` };
  }
}

function undoInfo(name, arg, text) {
  if (!text.startsWith("ok")) return null;
  if (name === "lembrar") return { tipo: "fato", ref: arg, rotulo: "desfazer" };
  if (name === "anotar") return { tipo: "nota", ref: arg, rotulo: "apagar" };
  if (name === "lembrete" && lastReminderId) return { tipo: "lembrete", ref: lastReminderId, rotulo: "cancelar" };
  return null;
}

// ============================================================ o laço (drive do web.py)
const STATE = { auto: store.get("lia-auto", false), permitidas: new Set(), temperature: store.get("lia-temp", 0.7) };
let pendente = null;  // chamada esperando confirmação
let ultimaFala = "";  // última fala da pessoa (o lembrete pega o horário dela se o modelo perder)

// Executa (ou cancela) a chamada pendente. Devolve [evento de resultado, eventos finais se a conversa para aqui].
async function executar(call, aprovado) {
  pendente = null; lastReminderId = null;
  if (!aprovado) return [{ type: "result", tool: call.tool, text: "cancelado: você não autorizou", undo: null }, null];
  if (SO_PC.has(call.tool) && !(call.tool === "abrir_app" && siteUrl(semEnchimento(call.arg)))) {
    const text = "erro: no navegador isso precisa do Lia Conector";
    await ultimo(pedir("anexar", { turns: [["result", text], ["assistant", TEXTO_SO_PC]] }));
    return [{ type: "result", tool: call.tool, text, undo: null },
      [{ type: "token", text: TEXTO_SO_PC }, { type: "text", text: TEXTO_SO_PC }]];
  }
  // "abre o youtube" pedido como app: abre o site
  const arg = call.tool === "lembrete" ? ancorarLembrete(call.arg, ultimaFala) : call.arg;
  const r = call.tool === "abrir_app" ? await rodarFerramenta("abrir_site", semEnchimento(arg))
    : await rodarFerramenta(call.tool, arg);
  return [{ type: "result", tool: call.tool, text: r.text, undo: undoInfo(call.tool, call.arg, r.text), link: r.link, titulo: r.titulo }, null];
}

async function* drive({ text, approve, retry }) {
  await pronta;
  await fila;
  let msg;
  if (approve !== undefined) {
    if (!pendente) return;
    const [res, fim] = await executar(pendente, approve);
    yield res;
    if (fim) { yield* fim; return; }
    msg = { type: "resultado", text: res.text };
  } else if (retry) {
    msg = { type: "retry" };
  } else {
    if (pendente) {  // mensagem nova com confirmação em aberto = cancelar
      await ultimo(pedir("anexar", { turns: [["result", "cancelado: você não autorizou"]] }));
      pendente = null;
    }
    msg = { type: "falar", text };
    ultimaFala = text;
  }
  for (let passo = 0; passo < 4; passo++) {
    let call = null;
    for await (const ev of pedir(msg.type, msg)) {
      if (ev.type === "fim") break;
      yield ev;
      if (ev.type === "call") call = ev;
    }
    if (!call) return;
    pendente = call;
    if (call.confirm && !STATE.auto && !STATE.permitidas.has(call.tool)) {
      yield { type: "confirm", tool: call.tool, arg: call.arg };
      return;
    }
    const [res, fim] = await executar(call, true);
    yield res;
    if (fim) { yield* fim; return; }
    msg = { type: "resultado", text: res.text };
  }
}

// ============================================================ endereços /api/...
const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
function ndjson(gen) {
  const enc = new TextEncoder();
  return new Response(new ReadableStream({
    async pull(ctrl) {
      const { value, done } = await gen.next();
      if (done) ctrl.close();
      else ctrl.enqueue(enc.encode(JSON.stringify(value) + "\n"));
    },
  }), { headers: { "Content-Type": "application/x-ndjson" } });
}

export async function handle(path, body) {
  const get = body === undefined, data = body || {};
  if (get && path === "/api/info") {
    const m = await manifestPronto;
    return json({
      params: m.params, iter: m.passo, val_loss: m.val_loss, auto: STATE.auto, temperature: STATE.temperature,
      apps: "no navegador", simulado: false,
      tools: Object.entries(m.ferramentas).map(([name, t]) => ({ name, confirm: t.confirm, desc: SO_PC.has(name) ? t.desc + " (só no PC, com o Conector)" : t.desc })),
    });
  }
  if (get && path === "/api/turns") return json({ turns: (await emFila(() => ultimo(pedir("turns")))).turns });
  if (get && path === "/api/memoria") return json(memoria());
  if (get && path === "/api/lembretes") return json(lembretesAtivos());
  if (get && path === "/api/events") return json(EVENTS.splice(0));
  if (get) return json({ erro: "não encontrado" }, 404);

  if (path === "/api/chat") {
    const text = String(data.text || "").trim().slice(0, 2000);
    if (!text) return json({ erro: "mensagem vazia" }, 400);
    return ndjson(drive({ text }));
  }
  if (path === "/api/confirm") {
    if (data.sempre && pendente) STATE.permitidas.add(pendente.tool);
    return ndjson(drive({ approve: !!data.approve }));
  }
  if (path === "/api/retry") return ndjson(drive({ retry: true }));
  if (path === "/api/perfil") {  // nome da boas-vindas vira memória
    const nome = String(data.nome || "").split(/\s+/).filter(Boolean).join(" ").slice(0, 40);
    if (nome) await rodarFerramenta("lembrar", `meu nome é ${nome}`);
    return json({ ok: !!nome });
  }
  if (path === "/api/desfazer") return json({ ok: true, msg: desfazer(String(data.tipo || ""), String(data.ref || "")) });
  if (path === "/api/reset") { pendente = null; emFila(() => ultimo(pedir("reset"))); return json({ ok: true }); }
  if (path === "/api/load") {
    pendente = null;
    const turns = Array.isArray(data.turns) ? data.turns : [];
    emFila(() => ultimo(pedir("load", { turns })));
    return json({ ok: true });
  }
  if (path === "/api/feedback") {  // "errou?": por enquanto fica só neste navegador
    const lista = store.get("lia-feedback", []);
    if (data.undo) lista.pop();
    else {
      const turns = (await emFila(() => ultimo(pedir("turns")))).turns;
      const item = { quando: new Date().toISOString(), origem: "site", messages: turns.map((t) => ({ role: t.role, content: t.text })) };
      if (String(data.esperado || "").trim()) item.esperado = String(data.esperado).trim().slice(0, 500);
      lista.push(item);
    }
    store.set("lia-feedback", lista.slice(-200));
    return json({ ok: true, total: lista.length });
  }
  if (path === "/api/memoria") {
    if (["fatos", "notas"].includes(data.tipo) && (data.apagar === null || Number.isInteger(data.apagar))) apagarMemoria(data.tipo, data.apagar);
    return json(memoria());
  }
  if (path === "/api/lembretes") { cancelarLembrete(String(data.cancelar || "")); return json(lembretesAtivos()); }
  if (path === "/api/settings") {
    if ("auto" in data) { STATE.auto = !!data.auto; store.set("lia-auto", STATE.auto); }
    if (Array.isArray(data.permitidas)) STATE.permitidas = new Set(data.permitidas.map(String));
    if ("temperature" in data) {
      STATE.temperature = Math.max(0, Math.min(1.5, Number(data.temperature) || 0));
      store.set("lia-temp", STATE.temperature);
      pronta.then(() => ultimo(pedir("config", { temperature: STATE.temperature }))).catch(() => {});
    }
    return json({ auto: STATE.auto, temperature: STATE.temperature, permitidas: [...STATE.permitidas].sort() });
  }
  return json({ erro: "não encontrado" }, 404);
}

