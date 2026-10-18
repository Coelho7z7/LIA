"""
Gera as respostas "certas" (PyTorch) para o teste de paridade do motor JS.
Lê os MESMOS pesos exportados (webapp/modelo), então qualquer diferença é do motor, não da exportação.

    python webapp/test/referencia.py              # -> webapp/test/referencia.json
    node webapp/test/paridade.js                  # compara
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from copia import Copia  # noqa: E402
from eval import predict  # noqa: E402
from make_dialogues import eval_cases  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

MODELO = ROOT / "webapp" / "modelo"

EXTRA_TEXTS = [
    "", " ", "  ", "\n", "a\n\nb", "oi  tudo bem?\t", "Ação, ÇÃO, coração!!!", "1234567 e 89,5% de 1.000",
    "e-mail: fulano_da.silva@exemplo.com", "C:\\Users\\Pasta\\arquivo.txt", "olá 😀 emoji 👍🏽 e símbolos ™ ® ©",
    "x² + y³ = ½", "ﬁm de ﬂuxo", "日本語のテキスト", "Привет мир", "tab\tno\tmeio", "fim com espaço ",
    "a\u00a0b", "a\u2003b", "a\x1fb", "a\x85b", "a\ufeffb", "cafe\u0301", "___ _x_ __init__", "!!??..,,;;",
    "abre o spotify pfv", "me lembra em 30 minutos de tirar o frango do forno, desligar o gas e chamar a galera",
]

PROMPTS = [  # (falas antes da vez da Lia)
    [("user", "oi lia, tudo bem?")],
    [("user", "abre o spotify")],
    [("user", "quanto é 37 vezes 48?"), ("call", "calcular: 37 vezes 48"), ("result", "ok: 1776")],
    [("user", "eu sou Matheus"), ("call", "lembrar: meu nome é Matheus"), ("result", "ok: guardei"),
     ("assistant", "Prazer, Matheus! Bora?"), ("user", "qual é o meu nome?")],
]


def carregar():
    man = json.loads((MODELO / "modelo.json").read_text(encoding="utf-8"))
    buf = b"".join((MODELO / c).read_bytes() for c in man["chunks"])
    sd = {}
    for t in man["tensores"]:
        n = int(np.prod(t["shape"]))
        if t["dtype"] == "f32":
            a = np.frombuffer(buf, "<f4", n, t["offset"])
        elif t["dtype"] == "f16":
            a = np.frombuffer(buf, "<f2", n, t["offset"]).astype(np.float32)
        else:
            rows = t["shape"][0]
            q = np.frombuffer(buf, np.int8, n, t["offset"]).astype(np.float32).reshape(rows, -1)
            a = q * np.frombuffer(buf, "<f4", rows, t["scales"])[:, None]
        sd[t["name"]] = torch.from_numpy(np.array(a, dtype=np.float32).reshape(t["shape"]))
    cfg = dict(man["config"], dropout=0.0)
    model = GPT(GPTConfig(**cfg)).eval()
    sd["lm_head.weight"] = sd["tok_emb.weight"]
    model.load_state_dict(sd)
    return man, model, BPETokenizer.from_dict(man["tokenizer"])


def ids_de(tok, falas):
    ids = [tok.bos]
    for role, text in falas:
        ids += tok.encode_turn(role, text)[0]
    return ids + [tok.assistant]


@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    man, model, tok = carregar()
    cases = eval_cases(40)
    ref = {"tokenizer": [], "logits": [], "chamadas": [], "gulosas": []}
    for text in EXTRA_TEXTS + [c["user"] for c in cases] + [c["content"] for c in cases]:
        ref["tokenizer"].append({"text": text, "ids": tok.encode(text)})

    for falas in PROMPTS:  # logits do prefill e de 3 passos com cache
        ids = ids_de(tok, falas)
        cache = model.new_cache()
        logits = model(torch.tensor([ids]), cache=cache)[0][0, -1]
        steps, passos = [], []
        for _ in range(3):
            t = int(logits.argmax())
            steps.append(t)
            logits_t = logits
            passos.append([round(float(v), 5) for v in logits_t])
            logits = model(torch.tensor([[t]]), cache=cache)[0][0, -1]
        passos.append([round(float(v), 5) for v in logits])
        ref["logits"].append({"ids": ids, "tokens": steps, "logits": passos})

    fixas = man["fixas"]
    for c in cases[::4]:  # chamadas com cópia forçada (o mesmo predict do eval.py)
        role, out = predict(model, tok, c["user"], fixas)
        ref["chamadas"].append({"user": c["user"], "role": role, "out": out})

    for falas in PROMPTS:  # resposta gulosa inteira (texto longo, várias falas no contexto)
        ids = ids_de(tok, falas)
        gen = list(model.generate(ids, 60, temperature=0, stop_ids=(tok.end,)))
        ref["gulosas"].append({"ids": ids, "gen": gen})

    out = Path(__file__).with_name("referencia.json")
    out.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    print(f"{out}: {len(ref['tokenizer'])} textos, {len(ref['logits'])} prompts, "
          f"{len(ref['chamadas'])} chamadas, {len(ref['gulosas'])} respostas gulosas")


if __name__ == "__main__":
    main()
