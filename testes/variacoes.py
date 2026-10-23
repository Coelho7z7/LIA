"""
Variações automáticas das frases soltas da bateria (testes/casos.py): o mesmo pedido escrito
de outros jeitos. Mede se a Lia aguenta o jeito de cada um escrever.

  v1  "do jeito do zap": minúsculo, sem acento, abreviado (vc, q, pq, hj, tb, pra)
  v2  com chamamento/pedido de favor ("lia, ...", "... pfv") e um erro de digitação

Nas variações a regra da ferramenta vale sem o argumento exato (só o nome da ferramenta),
porque abreviar ou errar letra muda o texto que ela copia. As regras de texto continuam.
"""

import random
import re
import unicodedata
import zlib

ABREV = [(r"\bvocê\b", "vc"), (r"\bque\b", "q"), (r"\bporque\b", "pq"), (r"\bpor que\b", "pq"), (r"\bhoje\b", "hj"),
         (r"\btambém\b", "tb"), (r"\bpara\b", "pra"), (r"\bpor favor\b", "pfv"), (r"\bquanto\b", "qnt"),
         (r"\bbeleza\b", "blz"), (r"\bobrigado\b", "obg"), (r"\btudo\b", "td"), (r"\bmesmo\b", "msm")]
CHAMA = ["lia, ", "ei lia, ", "lia ", "por favor, ", "então, ", "oi lia, "]
FECHA = [" pfv", " aí", " por favor", " rapidinho", " lia", ""]
PULAR = {"lixo", "contexto"}  # gibberish e conversas de vários turnos ficam como estão


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def zap(s):
    s = s.lower()
    for pat, rep in ABREV:
        s = re.sub(pat, rep, s)
    return sem_acento(s).rstrip("?!.")


def troca_letras(s, rng):
    ws = s.split(" ")
    cand = [i for i, w in enumerate(ws) if len(w) > 4 and w.isalpha()]
    if not cand:
        return s
    i = rng.choice(cand)
    w = ws[i]
    j = rng.randrange(len(w) - 1)
    ws[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(ws)


def solto(esperado):
    """'t:pesquisar=preco do iphone' -> 't:pesquisar' (mantém as regras de texto)."""
    regra, *texto = esperado.split(";")
    regra = "|".join(o.split("=")[0] for o in regra.split("|"))
    return ";".join([regra] + texto)


def variacoes(base):
    out = {}
    for cat, convs in base.items():
        if cat in PULAR:
            continue
        v1, v2 = [], []
        for conv in convs:
            if len(conv) != 1:
                continue
            frase, esperado, *resto = conv[0]
            rng = random.Random(zlib.crc32(frase.encode()))
            esp = solto(esperado)
            a = zap(frase)
            if a != frase.lower():
                v1.append([(a, esp, *resto)])
            conversa = esperado.split("|")[0].split(";")[0] == "c"
            chama, fecha = (["lia, ", "ei lia, ", "então, "], ["", " kkk", " lia"]) if conversa else (CHAMA, FECHA)
            pre = rng.choice(chama) if "lia" not in frase.lower() else ""
            fim = rng.choice(fecha)
            if "lia" in (pre + frase).lower() and fim == " lia":
                fim = ""
            b = troca_letras(pre + frase[:1].lower() + frase[1:] + fim, rng)
            v2.append([(b, esp, *resto)])
        if v1:
            out[f"{cat}~zap"] = v1
        out[f"{cat}~extra"] = v2
    return out
