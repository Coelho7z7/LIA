"""
Baixa resumos reais da Wikipédia em português para os exemplos de treino da ferramenta `consultar`.

    python baixar_wiki.py --aleatorios 4000      # -> data_wiki/artigos.jsonl

Pega os artigos da "Lista dos artigos que toda Wikipédia deve ter" (assuntos que a pessoa pergunta de
verdade) e mais artigos aleatórios (variedade de texto para ela aprender a copiar). Só o primeiro
parágrafo de cada um. Uma requisição por segundo, com User-Agent identificado, como a Wikimedia pede.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tools import WIKI_API, WIKI_UA

VITAIS = "Wikipédia:Lista dos artigos que toda Wikipédia deve ter"


def get(params):
    url = WIKI_API + "?" + urllib.parse.urlencode({"action": "query", "format": "json", "formatversion": 2, **params})
    for tentativa in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": WIKI_UA}), timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 5 * 2 ** tentativa))
        except OSError:
            time.sleep(5 * 2 ** tentativa)
    raise RuntimeError("a Wikipédia não respondeu")


def paginas(d):
    for p in d.get("query", {}).get("pages", []):
        texto = (p.get("extract") or "").strip()
        if texto and "pageprops" not in p:  # sem página de desambiguação
            yield {"titulo": p["title"], "texto": texto.split("\n")[0]}


EXTRATO = {"prop": "extracts|pageprops", "exintro": 1, "explaintext": 1, "exlimit": 20, "ppprop": "disambiguation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aleatorios", type=int, default=4000)
    ap.add_argument("--out", default="data_wiki/artigos.jsonl")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vistos = {}

    cont = {}
    while True:  # os vitais, 20 por vez
        d = get({"generator": "links", "titles": VITAIS, "gplnamespace": 0, "gpllimit": 20, **EXTRATO, **cont})
        for a in paginas(d):
            vistos[a["titulo"]] = a | {"fonte": "vital"}
        if "continue" not in d:
            break
        cont = d["continue"]
        time.sleep(1)
    print(f"vitais: {len(vistos)}")

    n0 = len(vistos)
    while len(vistos) - n0 < args.aleatorios:
        for a in paginas(get({"generator": "random", "grnnamespace": 0, "grnlimit": 20, **EXTRATO})):
            vistos.setdefault(a["titulo"], a | {"fonte": "aleatorio"})
        time.sleep(1)
    with open(out, "w", encoding="utf-8") as f:
        for a in vistos.values():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"{out}: {len(vistos)} artigos ({len(vistos) - n0} aleatórios)")


if __name__ == "__main__":
    main()
