"""
Pipeline completo com um comando:
  1. gera as conversas sintéticas (se ainda não existirem)
  2. tokeniza tudo (prepare.py)
  3. pré-treina nos seus textos, se houver .txt em data/textos/
  4. treina o chat (SFT)
  5. mede o resultado (eval.py)

    python treinar.py                                # padrão: ~90 min de chat
    python treinar.py --nome Nina --minutos-chat 120
    python treinar.py --minutos-pretreino 120 --minutos-chat 60 --preset medium
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(*args):
    print(f"\n$ python {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, *args])
    if r.returncode != 0:
        raise SystemExit(f"Falhou: {' '.join(args)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", default="Lia", help="nome da IA")
    ap.add_argument("--conversas", type=int, default=30000)
    ap.add_argument("--regerar", action="store_true", help="gera as conversas sintéticas de novo")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--minutos-pretreino", type=float, default=60)
    ap.add_argument("--minutos-chat", type=float, default=90)
    args = ap.parse_args()

    synth = Path("data/conversas/sinteticas.jsonl")
    if args.regerar or not synth.exists():
        run("make_dialogues.py", "--n", str(args.conversas), "--nome", args.nome, "--out", str(synth))
    run("prepare.py", "--vocab", str(args.vocab))
    if Path("out/pretrain_train.bin").exists():
        run("train.py", "--stage", "pretrain", "--preset", args.preset, "--max-iters", "100000",
            "--max-minutes", str(args.minutos_pretreino))
    else:
        Path("out/base.pt").unlink(missing_ok=True)  # base antigo com vocabulário velho não serve mais
        print("\n(sem .txt em data/textos/: pulando o pré-treino)")
    run("train.py", "--stage", "sft", "--preset", args.preset, "--max-iters", "100000",
        "--max-minutes", str(args.minutos_chat), "--eval-interval", "150")
    run("eval.py")
    print("\nPronto! Converse com: python web.py   (ou python chat.py)")


if __name__ == "__main__":
    main()
