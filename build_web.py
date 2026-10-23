"""
Monta o site da Lia (webapp/dist) a partir da interface local (web/index.html) + motor JS.

    python export_web.py --ckpt out_v9/chat.pt     # pesos -> webapp/modelo
    python build_web.py                             # site -> webapp/dist
    npx wrangler pages deploy webapp/dist --project-name lia-ai

A interface é a mesma do web.py: só a função api() muda, para falar com o backend.js
(o "servidor" que roda dentro da própria página) em vez do Python.
"""

import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB, APP = ROOT / "web", ROOT / "webapp"
DIST = APP / "dist"
LIMIT = 25 * 1024 * 1024  # Cloudflare Pages: até 25 MiB por arquivo

PATCHES = [  # (trecho do web/index.html, troca na versão web)
    ('const TOKEN = "__TOKEN__";',
     '// versão web: o "servidor" é o backend.js, que roda dentro da própria página\n'
     'const LIA_BACKEND = import("./backend.js");'),
    ('''function api(path, body) {
  return fetch(path, { method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "X-Token": TOKEN },
    body: body === undefined ? undefined : JSON.stringify(body) });
}''', '''function api(path, body) {
  return LIA_BACKEND.then((b) => b.handle(path, body));
}'''),
    ("`${(info.params / 1e6).toFixed(1)}M parâmetros · ${info.apps} apps`",
     "`${(info.params / 1e6).toFixed(1)}M parâmetros · ${info.apps}`"),
    ('"Perdi a conexão com o servidor (" + e.message + "). Ele ainda tá rodando?"',
     '"Deu um erro aqui (" + e.message + "). Recarrega a página?"'),
    ("Roda inteiro neste computador", "Roda inteiro no seu navegador, neste computador"),
    ('<div class="about" id="about">carregando...</div>',
     '<div class="about" id="about">carregando...</div>\n'
     '  <div class="about"><a href="/privacidade.html" target="_blank" rel="noopener">Privacidade</a></div>'),
]

COPY = ["backend.js", "ferramentas.js", "sw.js", "_headers", "privacidade.html",
        "engine/lia.js", "engine/model.js", "engine/tokenizer.js", "engine/copia.js", "engine/worker.js"]


def main():
    man_path = APP / "modelo" / "modelo.json"
    if not man_path.exists():
        raise SystemExit("Falta webapp/modelo: rode antes  python export_web.py --ckpt out_XX/chat.pt")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "engine").mkdir(parents=True)
    (DIST / "modelo").mkdir()

    html = (WEB / "index.html").read_text(encoding="utf-8")
    for a, b in PATCHES:
        assert html.count(a) == 1, f"trecho não achado (ou repetido) no web/index.html: {a[:60]!r}"
        html = html.replace(a, b)
    (DIST / "index.html").write_text(html, encoding="utf-8")

    for f in COPY:
        shutil.copy2(APP / f, DIST / f)
    for f in ["modelo.json"] + man["chunks"]:
        shutil.copy2(APP / "modelo" / f, DIST / "modelo" / f)
    for f in ["icon-192.png", "icon-512.png"]:
        shutil.copy2(WEB / f, DIST / f)
    manifest = json.loads((WEB / "manifest.json").read_text(encoding="utf-8"))
    manifest["description"] = "IA pessoal treinada do zero, rodando no seu navegador"
    (DIST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # versão do service worker: muda a cada build, então todo mundo recebe a casca nova
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(DIST.rglob("*")) if p.is_file()
                                     and p.suffix != ".bin")).hexdigest()[:8]
    build = f"{time.strftime('%Y%m%d%H%M')}-{digest}"
    sw = DIST / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace("__BUILD__", build), encoding="utf-8")

    files = [p for p in DIST.rglob("*") if p.is_file()]
    grandes = [p for p in files if p.stat().st_size > LIMIT]
    if grandes:
        raise SystemExit(f"Arquivo maior que 25 MiB: {grandes}")
    total = sum(p.stat().st_size for p in files)
    print(f"{DIST}: {len(files)} arquivos, {total / 2**20:.1f} MiB | modelo passo {man['passo']} ({man['origem']}) | build {build}")


if __name__ == "__main__":
    main()
