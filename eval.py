"""
Mede o quanto o modelo acerta em pedidos que NUNCA viu no treino (frases, apps e
termos reservados para teste pelo make_dialogues.py).

    python eval.py
    python eval.py --n 60 --erros 20

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
from make_dialogues import eval_cases
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


@torch.no_grad()
def predict(model, tok, user_text):
    """Primeira fala do assistente: ("call", "nome: arg") ou ("assistant", None)."""
    ids = [tok.bos] + tok.encode_turn("user", user_text)[0] + [tok.assistant]
    dev = model.tok_emb.weight.device
    cache = model.new_cache()
    logits, _ = model(torch.tensor([ids], device=dev), cache=cache)
    logits = logits[0, -1]
    if int(logits.argmax()) != tok.call:
        return "assistant", None
    gen, t = [], tok.call
    for _ in range(48):
        logits, _ = model(torch.tensor([[t]], device=dev), cache=cache)
        t = int(logits[0, -1].argmax())
        if t == tok.end:
            break
        gen.append(t)
    return "call", tok.decode(gen).strip()


@torch.no_grad()
def reply_after(model, tok, user_text, call, result):
    """Resposta do assistente depois de ver o resultado da ferramenta (gulosa)."""
    ids = [tok.bos] + tok.encode_turn("user", user_text)[0] + tok.encode_turn("call", call)[0]
    ids += tok.encode_turn("result", result)[0] + [tok.assistant]
    return tok.decode(list(model.generate(ids, 80, temperature=0, stop_ids=(tok.end,))))


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
    return None


def same_arg(a, b):
    return tools.norm(a or "") == tools.norm(b or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--n", type=int, default=40, help="casos por tipo de pedido")
    ap.add_argument("--erros", type=int, default=12, help="quantos erros mostrar")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tok = BPETokenizer.from_dict(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["config"])).eval()
    model.load_state_dict(ckpt["model"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modelo: {args.ckpt} (passo {ckpt['iter']}, val loss {ckpt['best_val']:.3f})\n")

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
        role, out = predict(model, tok, c["user"])
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
