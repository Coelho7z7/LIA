"""
Continua um texto com o modelo base (pré-treinado nos seus textos).

    python generate.py --prompt "Era uma vez"
    python generate.py                     # modo interativo

O modelo base não conversa nem usa ferramentas: ele CONTINUA o que você começar,
no estilo dos seus textos. Para conversar, use chat.py ou web.py.
"""

import argparse
import codecs
import sys

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/base.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8, help="maior = mais criativo/caótico")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    tok = BPETokenizer.from_dict(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])

    def run(prompt):
        ids = [tok.bos] + tok.encode(prompt)
        # Um caractere acentuado pode vir dividido em dois tokens; o decodificador
        # incremental segura os bytes até o caractere ficar completo.
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        print(prompt, end="", flush=True)
        for t in model.generate(ids, args.tokens, args.temperature, args.top_k, args.top_p,
                                stop_ids=(tok.bos, tok.end)):
            print(dec.decode(tok.vocab[t]), end="", flush=True)
        print(dec.decode(b"", final=True))

    if args.prompt is not None:
        run(args.prompt)
        return
    print(f"Modelo de {args.ckpt} (passo {ckpt['iter']}). Ctrl+C para sair.\n")
    try:
        while True:
            run(input("> "))
            print()
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
