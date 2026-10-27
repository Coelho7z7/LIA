"""
Mede o quanto o modelo acerta em pedidos que NUNCA viu no treino (frases, apps e
termos reservados para teste pelo make_dialogues.py).

    python eval.py
    python eval.py --n 60 --erros 20
    python eval.py --conselho      # a primeira fala passa pelo conselho (conselho.py)

Para cada pedido, confere:
  rota       decidiu certo entre conversar e usar ferramenta?
  ferramenta escolheu a ferramenta certa?
  argumento  copiou o argumento certo (ex.: o nome do app)?
"""

import argparse
import sys
from collections import defaultdict

import torch

import tools
from agent import TRAVA
from conselho import Conselho
from copia import Copia, carregar_fixas
from make_dialogues import eval_cases
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


@torch.no_grad()
def predict(model, tok, user_text, fixas=None):
    """Primeira fala do assistente: ("call", "nome: arg") ou ("assistant", None).
    Com `fixas`, usa a mesma cópia forçada do agente (copia.py)."""
    ids = [tok.bos] + tok.encode_turn("user", user_text)[0] + [tok.assistant]
    dev = model.tok_emb.weight.device
    cache = model.new_cache()
    logits, _ = model(torch.tensor([ids], device=dev), cache=cache)
    logits = logits[0, -1]
    if int(logits.argmax()) != tok.call:
        return "assistant", None
    copia = Copia(fixas, user_text) if fixas else None
    gen, raw, t = [], b"", tok.call
    for _ in range(48):
        logits, _ = model(torch.tensor([[t]], device=dev), cache=cache)
        t = copia.escolhe(logits[0, -1], raw, tok) if copia else int(logits[0, -1].argmax())
        if t is None or t == tok.end:
            break
        gen.append(t)
        raw += tok.vocab[t]
    return "call", tok.decode(gen).strip()


@torch.no_grad()
def predict_conselho(conselho, model, tok, user_text):
    """Igual a predict(), mas quem decide é o conselho."""
    ids = [tok.bos] + tok.encode_turn("user", user_text)[0] + [tok.assistant]
    cache = model.new_cache()
    logits, _ = model(torch.tensor([ids], device=model.tok_emb.weight.device), cache=cache)
    win, _, _ = conselho.decidir(cache, logits[0, -1], user_text, trava=TRAVA)  # a mesma trava do agente
    return ("call", win.text) if win.kind == "call" else ("assistant", None)


@torch.no_grad()
def reply_after(model, tok, user_text, call, result):
    """Resposta do assistente depois de ver o resultado da ferramenta (gulosa)."""
    ids = [tok.bos] + tok.encode_turn("user", user_text)[0] + tok.encode_turn("call", call)[0]
    ids += tok.encode_turn("result", result)[0] + [tok.assistant]
    return tok.decode(list(model.generate(ids, 160, temperature=0, stop_ids=(tok.end,))))  # resumo da Wikipédia é longo


def key_value(kind, result):
    """O pedaço do resultado que a resposta TEM que repetir (ex.: a hora certa)."""
    if not result.startswith("ok: ") or "nenhuma nota" in result:
        return None
    v = result[4:]
    if kind == "app":
        return v.removeprefix("abri ")
    if kind == "remind":
        return v.split(":")[0].removeprefix("em ")
    if kind in ("hora", "data", "calc", "weather", "recall", "see_notes"):
        return v
    if kind == "factual" and not v.startswith("pesquisa aberta"):  # consultar: a resposta repete o resumo
        return v
    return None


def same_arg(a, b):
    return tools.norm(a or "") == tools.norm(b or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--n", type=int, default=40, help="casos por tipo de pedido")
    ap.add_argument("--erros", type=int, default=12, help="quantos erros mostrar")
    ap.add_argument("--conselho", action="store_true", help="decide a primeira fala com o conselho")
    ap.add_argument("--sem-copia", action="store_true", help="desliga a cópia forçada (mede o modelo puro)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    tok = BPETokenizer.from_dict(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["config"])).eval()
    model.load_state_dict(ckpt["model"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    fixas = None if args.sem_copia else carregar_fixas(args.ckpt)
    print(f"Modelo: {args.ckpt} (passo {ckpt['iter']}, val loss {ckpt['best_val']:.3f}) | "
          f"cópia forçada: {'ligada' if fixas else 'desligada'}" + (" | com conselho" if args.conselho else "") + "\n")
    conselho = None
    if args.conselho:
        conselho = Conselho(model, tok, fixas=fixas)
        torch.manual_seed(0)  # o conselho sorteia propostas: mesma semente, mesmo resultado

    stats = defaultdict(lambda: [0, 0, 0, 0])  # total, rota, ferramenta, argumento
    faithful = defaultdict(lambda: [0, 0])  # respostas que repetem o valor certo / total
    errors, unfaithful = [], []
    for c in eval_cases(args.n):
        key = key_value(c["kind"], c.get("result", ""))
        if key:
            reply = reply_after(model, tok, c["user"], c["content"], c["result"])
            hit = tools.norm(key) in tools.norm(reply)
            faithful[c["kind"]][0] += hit
            faithful[c["kind"]][1] += 1
            if not hit:
                unfaithful.append((c["result"], reply))
        role, out = (predict_conselho(conselho, model, tok, c["user"]) if conselho
                     else predict(model, tok, c["user"], fixas))
        s = stats[c["kind"]]
        s[0] += 1
        route = role == c["role"]
        if c["role"] == "assistant":
            ok_tool = ok_arg = route
        else:
            exp_name, _, exp_arg = c["content"].partition(":")
            got_name, _, got_arg = (out or "").partition(":")
            ok_tool = route and got_name.strip() == exp_name.strip()
            ok_arg = ok_tool and same_arg(got_arg, exp_arg)
        s[1] += route
        s[2] += ok_tool
        s[3] += ok_arg
        if not ok_arg:
            errors.append((c["kind"], c["user"], c["content"] if c["role"] == "call" else "(conversa)",
                           out if role == "call" else "(conversa)"))

    print(f"{'tipo':<11} {'casos':>5} {'rota':>7} {'ferram.':>8} {'argum.':>7} {'fidelid.':>9}")
    tot = [0, 0, 0, 0]
    for kind, s in stats.items():
        tot = [a + b for a, b in zip(tot, s)]
        f = faithful.get(kind)
        fid = f"{f[0] / f[1]:>9.0%}" if f and f[1] else f"{'-':>9}"
        print(f"{kind:<11} {s[0]:>5} {s[1] / s[0]:>7.0%} {s[2] / s[0]:>8.0%} {s[3] / s[0]:>7.0%} {fid}")
    fh, ft = sum(f[0] for f in faithful.values()), sum(f[1] for f in faithful.values())
    print(f"{'TOTAL':<11} {tot[0]:>5} {tot[1] / tot[0]:>7.0%} {tot[2] / tot[0]:>8.0%} {tot[3] / tot[0]:>7.0%} "
          f"{fh / max(ft, 1):>9.0%}")
    print("\nrota = conversar x usar ferramenta | fidelid. = a resposta repete o valor que a ferramenta devolveu")
    if unfaithful and args.erros:
        print(f"\nRespostas infiéis ao resultado ({len(unfaithful)}):")
        for res, rep in unfaithful[:max(3, args.erros // 3)]:
            print(f"  resultado: {res}\n  resposta:  {rep}")

    if errors and args.erros:
        print(f"\nAlguns erros ({len(errors)} no total):")
        for kind, user, exp, got in errors[::max(1, len(errors) // args.erros)][:args.erros]:
            print(f"  [{kind}] {user!r}\n      esperado: {exp}\n      saiu:     {got}")


if __name__ == "__main__":
    main()
