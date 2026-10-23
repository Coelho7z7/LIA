"""
Cópia forçada na chamada de ferramenta.

Quase todo argumento é um pedaço do que a pessoa disse ("abre o PowerPoint" ->
"abrir_app: PowerPoint"). Modelo pequeno às vezes erra a cópia letra a letra
("Power", "remio", "quem ganou"). Aqui, a cada token da chamada, o agente pega o
token mais provável que mantém o argumento válido:

  - cada palavra vem da conversa (fala da pessoa, resultados, respostas) ou do
    vocabulário fixo da ferramenta ("meu nome é", "mais", "proxima"...);
  - uma palavra da conversa aparece no máximo uma vez a mais do que foi dita
    (barra laços tipo "de de de de");
  - pontuação só a que já está na conversa (mais espaço, vírgula e "|").

O vocabulário fixo sai dos dados de treino (prepare.py grava fixas.json ao lado
do modelo). Sem fixas.json, a geração fica como antes.

    python copia.py data_v9/conversas out_v9    # gera out_v9/fixas.json
"""

import codecs
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

W = re.compile(r"\w+")
EXTRA_SYMBOLS = set(" ,|")


def palavras(s):
    return W.findall(s.lower())


def fixas_das_conversas(convs, min_frac=2e-4):
    """Palavras que o argumento de cada ferramenta usa sem estarem na conversa (comuns no treino)."""
    fixed, n = defaultdict(Counter), 0
    for msgs in convs:
        n += 1
        seen = set()  # tudo que a pessoa e as ferramentas já disseram até ali
        for a, b in zip(msgs, msgs[1:]):
            if a["role"] in ("user", "result"):
                seen |= set(palavras(a["content"]))
            if a["role"] == "user" and b["role"] == "call":
                tool, _, arg = b["content"].partition(":")
                fixed[tool.strip()].update(w for w in palavras(arg) if w not in seen)
    lim = max(5, n * min_frac)
    out = {t: sorted(w for w, k in c.items() if k >= lim) for t, c in fixed.items()}
    return {t: ws for t, ws in out.items() if ws}


def carregar_fixas(ckpt_path):
    p = Path(ckpt_path).with_name("fixas.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


class Copia:
    def __init__(self, fixas, contexto):
        self.fixas = {t: set(ws) for t, ws in fixas.items()}
        self.saldo = Counter({w: n + 1 for w, n in Counter(palavras(contexto)).items()})
        self.simbolos = set(W.sub("", contexto.lower())) | EXTRA_SYMBOLS

    def aceita(self, raw, final=False):
        """`raw` são os bytes da chamada até aqui ("abrir_app: Pow..."). final=True: pode terminar aqui?"""
        try:  # acento pela metade só no fim; sequência inválida reprova
            text, n = codecs.utf_8_decode(raw, "strict", final)
        except UnicodeDecodeError:
            return False
        tail = raw[n:]
        if ":" not in text:  # ainda no nome da ferramenta
            return not final
        tool, _, arg = text.partition(":")
        fixed = self.fixas.get(tool.strip(), set())
        arg = arg.lower()
        if set(W.sub("", arg)) - self.simbolos:
            return False
        ws = W.findall(arg)
        if not ws and not tail:
            return True
        ends_word = arg[-1:].isalnum() or arg[-1:] == "_"
        if tail:  # a última palavra termina num acento pela metade
            done, last = (ws[:-1], ws[-1].encode() + tail) if ends_word else (ws, tail)
        elif ends_word and not final:
            done, last = ws[:-1], ws[-1].encode()
        else:
            done, last = ws, None
        # Palavra fixa da ferramenta só vale no começo ("meu nome é", "eu trabalho como"); depois que a cópia da
        # conversa começou, toda palavra tem que vir da conversa ("analista de moro" não passa).
        used, copiando = Counter(), False
        for w in done:
            if w in fixed and not copiando:
                continue
            used[w] += 1
            copiando = copiando or w not in fixed
            if used[w] > self.saldo.get(w, 0):
                return False
        if last is None:
            return True
        # a palavra aberta tem que poder virar uma da conversa que ainda tem saldo (ou fixa, se ainda no começo)
        return (not copiando and any(w.encode().startswith(last) for w in fixed)) or \
            any(w.encode().startswith(last) for w, k in self.saldo.items() if k > used[w])

    def escolhe(self, logits, gen, tok):
        """O token mais provável que mantém a chamada válida (None se nenhum serve)."""
        n_tok = 256 + len(tok.merges)
        for c in logits.argsort(descending=True).tolist():
            if c == tok.end:
                ok = self.aceita(gen, final=True)
            elif c >= n_tok:  # especiais (fora o fim) nunca entram numa chamada
                continue
            else:
                ok = self.aceita(gen + tok.vocab[c])
            if ok:
                return c
        return None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pasta, out = Path(sys.argv[1]), Path(sys.argv[2])
    convs = (json.loads(line)["messages"] for f in sorted(pasta.rglob("*.jsonl"))
             for line in open(f, encoding="utf-8") if line.strip())
    fixas = fixas_das_conversas(convs)
    (out / "fixas.json").write_text(json.dumps(fixas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out / 'fixas.json'}: " + ", ".join(f"{t} ({len(ws)})" for t, ws in fixas.items()))
