// Ferramentas em JS x tools.py (ferramentas.json):   node webapp/test/ferramentas.js
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as F from "../ferramentas.js";

const ref = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "ferramentas.json"), "utf8"));
const calc = (e) => { try { return "ok: " + F.calcularValor(e); } catch { return "erro: não consegui calcular isso"; } };
const dur = (d) => { try { return F.parseDuracao(d); } catch { return null; } };
const checks = {
  calcular: calc, duracao: dur, formatar_duracao: F.formatarDuracao, site_url: F.siteUrl,
  recordar: (q) => F.recordarEm(ref.fatos, q), norm: F.norm,
  horario: (e) => {
    const [ag, h] = e.split("|"), [hh, mm] = ag.split(":").map(Number);
    try { return F.parseDuracao(h, new Date(2026, 9, 17, hh, mm)); } catch { return null; }
  },
  resumo_wiki: (t) => F.resumoWiki(t), assunto_wiki: (a) => F.assuntoWiki(a),
  ancorar: (e) => { const i = e.indexOf("|", e.indexOf("|") >= 0 && e.split("|").length > 2 ? e.indexOf("|") + 1 : 0); return F.ancorarLembrete(e.slice(0, i), e.slice(i + 1)); },
};
let todas = true;
for (const [nome, fn] of Object.entries(checks)) {
  let ok = 0, shown = 0;
  for (const [entrada, esperado] of ref[nome]) {
    const got = fn(entrada);
    if (got === esperado) ok++;
    else if (shown++ < 6) console.log(`  ${nome}(${JSON.stringify(entrada)}): py ${JSON.stringify(esperado)} | js ${JSON.stringify(got)}`);
  }
  console.log(`${nome}: ${ok}/${ref[nome].length} iguais`);
  if (ok !== ref[nome].length) todas = false;
}
console.log(todas ? "FERRAMENTAS OK" : "FERRAMENTAS COM DIFERENÇAS");
process.exit(todas ? 0 : 1);
