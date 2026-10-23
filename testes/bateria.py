"""
Roda a bateria de uso real (testes/casos.py) numa versão da Lia e anota tudo.

    python testes/bateria.py --ckpt out_v9/chat.pt              # -> testes/resultados/out_v9.jsonl e .txt
    python testes/bateria.py --ckpt out_v9/chat.pt --cpu        # sem atrapalhar um treino na GPU
    python testes/bateria.py --comparar out_v9 out_v10          # o que melhorou e o que piorou

Modo simulado: nada é aberto de verdade e a memória usada é temporária (a sua não é tocada).
As respostas em texto são sorteadas como no uso normal, com semente fixa por frase (dá para repetir).
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools  # noqa: E402
from casos import CATEGORIAS as BASE  # noqa: E402
from conversas import CATEGORIAS as CONVERSAS  # noqa: E402
from variacoes import variacoes  # noqa: E402

CATEGORIAS = {**BASE, **CONVERSAS, **variacoes(BASE)}

RES = Path(__file__).resolve().parent / "resultados"


def confere_texto(regras, reply):
    """";tem=a/b" (alguma aparece na resposta) e ";nao=x/y" (nenhuma aparece), sem acento/maiúscula."""
    r = tools.norm(reply)
    for regra in regras:
        tipo, _, palavras = regra.partition("=")
        ws = [tools.norm(w) for w in palavras.split("/") if w.strip()]
        if tipo == "tem" and not any(w in r for w in ws):
            return False
        if tipo == "nao" and any(w in r for w in ws):
            return False
    return True


def confere(esperado, calls, reply=""):
    """True/False conforme a regra do caso; None quando é só para observar."""
    esperado, *regras = esperado.split(";")
    if regras and not confere_texto(regras, reply):
        return False
    if esperado == "?":
        return None
    for opt in esperado.split("|"):
        if opt == "?":
            return None if not calls else True
        if opt == "c" and not calls:
            return True
        if opt.startswith("t:") and calls:
            tool, _, arg = opt[2:].partition("=")
            if calls[0]["tool"] == tool and (not arg or tools.norm(calls[0]["arg"]) == tools.norm(arg)):
                return True
    return False


def rodar(ckpt, cpu, trava=0, porteiro=None):
    from agent import Assistant
    tools.DRY_RUN = True
    mem = RES / "_memoria_teste"
    tools.MEMORY_DIR = mem
    from trava import aplicar
    bot = aplicar(Assistant(str(ROOT / ckpt), device="cpu" if cpu else None, porteiro=porteiro), trava)
    nome = Path(ckpt).parent.name + ("_trava" if trava else "") + ("_porteiro" if porteiro else "")
    RES.mkdir(exist_ok=True)
    out, log = [], []
    k = 0
    for cat, convs in CATEGORIAS.items():
        for ci, conv in enumerate(convs):
            bot.reset()
            shutil.rmtree(mem, ignore_errors=True)  # cada conversa começa sem memória
            log.append(f"--- [{cat} {ci}]")
            for ti, turno in enumerate(conv):
                frase, esperado = turno[0], turno[1]
                nota = turno[2] if len(turno) > 2 else ""
                torch.manual_seed(1000 + k)
                k += 1
                calls, reply = [], ""
                for ev in bot.chat(frase, auto=True):
                    if ev["type"] == "call":
                        calls.append({"tool": ev["tool"], "arg": ev["arg"], "result": ""})
                    elif ev["type"] == "result" and calls:
                        calls[-1]["result"] = ev["text"]
                    elif ev["type"] == "text":
                        reply = ev["text"]
                ok = confere(esperado, calls, reply)
                out.append({"cat": cat, "conv": ci, "turno": ti, "user": frase, "esperado": esperado, "nota": nota,
                            "calls": calls, "reply": reply, "ok": ok})
                marca = {True: "  ", False: "XX", None: "??"}[ok]
                log.append(f"{marca} você: {frase}   [esperado: {esperado}{' | ' + nota if nota else ''}]")
                for c in calls:
                    log.append(f"     [{c['tool']}: {c['arg']}] -> {c['result']}")
                log.append(f"     lia: {reply}")
    shutil.rmtree(mem, ignore_errors=True)
    (RES / f"{nome}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    (RES / f"{nome}.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    resumo(out, nome)


def resumo(out, nome):
    por = defaultdict(lambda: [0, 0, 0])  # julgados, certos, só observar
    for r in out:
        p = por[r["cat"]]
        if r["ok"] is None:
            p[2] += 1
        else:
            p[0] += 1
            p[1] += r["ok"]
    print(f"\nBateria de uso real: {nome}\n{'categoria':<18} {'certos':>9} {'%':>5}  observar")
    tj = tc = 0
    for cat, (j, c, o) in por.items():
        tj, tc = tj + j, tc + c
        print(f"{cat:<18} {c:>4}/{j:<4} {c / max(j, 1):>5.0%}  {o}")
    print(f"{'TOTAL':<18} {tc:>4}/{tj:<4} {tc / max(tj, 1):>5.0%}")


def comparar(a, b):
    ra = {(r["cat"], r["conv"], r["turno"]): r for r in map(json.loads, open(RES / f"{a}.jsonl", encoding="utf-8"))}
    rb = [json.loads(l) for l in open(RES / f"{b}.jsonl", encoding="utf-8")]
    resumo(list(ra.values()), a)
    resumo(rb, b)
    for r in rb:
        o = ra.get((r["cat"], r["conv"], r["turno"]))
        if o and o["ok"] is not None and r["ok"] is not None and o["ok"] != r["ok"]:
            calls = "; ".join(f"{c['tool']}: {c['arg']}" for c in r["calls"]) or "(conversa)"
            print(f"{'MELHOROU' if r['ok'] else 'PIOROU  '} [{r['cat']}] {r['user']!r} -> {calls}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--trava", type=float, default=0, help="trava de confiança (ex.: 0.9); ver testes/trava.py")
    ap.add_argument("--porteiro", default=None, help="caminho do porteiro.pt (ver porteiro.py)")
    ap.add_argument("--comparar", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.comparar:
        comparar(*args.comparar)
    else:
        rodar(args.ckpt, args.cpu, args.trava, args.porteiro)


if __name__ == "__main__":
    main()
