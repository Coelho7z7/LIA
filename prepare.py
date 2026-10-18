"""
Prepara os dados de treino:
  1. Treina o tokenizador BPE em TODO o texto (seus textos + conversas).
  2. Pré-treino: data/textos/**/*.txt -> out/pretrain_{train,val}.bin
  3. Chat (SFT): data/conversas/**/*.jsonl -> out/sft_{train,val}.bin + máscara

    python prepare.py
    python prepare.py --vocab 8192      # vocabulário maior se tiver muito texto

Os .txt são opcionais. Sem eles, o modelo aprende só com as conversas.
"""

import argparse
import json
import random
import sys
import time
from array import array
from pathlib import Path

import numpy as np

from copia import fixas_das_conversas
from tokenizer import BPETokenizer


def load_texts(folder):
    files = sorted(Path(folder).rglob("*.txt")) if Path(folder).exists() else []
    return [f.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n") for f in files], files


def load_conversations(folder):
    convs = []
    for f in sorted(Path(folder).rglob("*.jsonl")) if Path(folder).exists() else []:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                print(f"  aviso: {f.name}:{n} não é JSON válido, pulando")
                continue
            msgs = obj["messages"] if "messages" in obj else [
                {"role": "user", "content": obj["user"]}, {"role": "assistant", "content": obj["assistant"]}]
            convs.append(msgs)
    return convs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--textos", default="data/textos", help="pasta com .txt para o pré-treino")
    ap.add_argument("--conversas", default="data/conversas", help="pasta com .jsonl de conversas")
    ap.add_argument("--out", default="out")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bpe-dropout", type=float, default=0.0,
                    help="chance de pular cada fusão do BPE no treino (ex.: 0.1); ensina os pedaços raros")
    ap.add_argument("--dropout-frac", type=float, default=0.6, help="fração das conversas de treino com dropout")
    ap.add_argument("--tokenizer", default=None,
                    help="usa um tokenizador pronto (tokenizer.json ou checkpoint .pt) em vez de treinar outro: "
                         "obrigatório para ajustar um modelo que já existe (train.py --init) ou partir de um pré-treino")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    texts, files = load_texts(args.textos)
    convs = load_conversations(args.conversas)
    print(f"Textos: {len(files)} arquivo(s), {sum(map(len, texts)):,} caracteres")
    print(f"Conversas: {len(convs):,}")
    if not texts and not convs:
        raise SystemExit("Nada para treinar. Rode make_dialogues.py e/ou coloque .txt em data/textos/")

    t0 = time.time()
    if args.tokenizer:
        if args.tokenizer.endswith(".pt"):
            import torch
            tok = BPETokenizer.from_dict(torch.load(args.tokenizer, map_location="cpu", weights_only=False)["tokenizer"])
        else:
            tok = BPETokenizer.load(args.tokenizer)
        print(f"Tokenizador pronto: {args.tokenizer}")
    else:
        corpus = "\n".join(texts + [m["content"] for c in convs for m in c])
        tok = BPETokenizer()
        print(f"Treinando tokenizador (vocab={args.vocab})...")
        tok.train(corpus, args.vocab)
        print(f"  {tok.vocab_size} tokens em {time.time() - t0:.1f}s")
    tok.save(out / "tokenizer.json")
    meta = {"vocab_size": tok.vocab_size}

    if texts:
        docs = [[tok.bos] + tok.encode(t) for t in texts]
        ids = np.array([i for d in docs for i in d], dtype=np.uint16)
        n_val = max(int(len(ids) * 0.1), 1024)
        ids[:-n_val].tofile(out / "pretrain_train.bin")
        ids[-n_val:].tofile(out / "pretrain_val.bin")
        meta["pretrain_tokens"] = int(len(ids))
        print(f"Pré-treino: {len(ids):,} tokens ({sum(map(len, texts)) / len(ids):.2f} caracteres/token)")
    else:
        for name in ("pretrain_train.bin", "pretrain_val.bin"):
            (out / name).unlink(missing_ok=True)

    if convs:
        random.Random(args.seed).shuffle(convs)
        n_val = max(int(len(convs) * args.val_frac), min(50, len(convs) // 5))
        rng = random.Random(args.seed + 1)
        for split, part in (("train", convs[n_val:]), ("val", convs[:n_val])):
            ids, mask, starts = array("H"), array("B"), []  # arrays compactos: cabe milhões de conversas na RAM
            for c in part:
                drop = args.bpe_dropout if split == "train" and rng.random() < args.dropout_frac else 0.0
                i, m = tok.encode_chat(c, drop, rng)
                starts.append(len(ids))
                ids.extend(i)
                mask.extend(m)
            np.frombuffer(ids, dtype=np.uint16).tofile(out / f"sft_{split}.bin")
            np.frombuffer(mask, dtype=np.uint8).tofile(out / f"sft_{split}_mask.bin")
            np.save(out / f"sft_{split}_starts.npy", np.array(starts, dtype=np.int64))
            meta[f"sft_{split}_tokens"] = len(ids)
            meta[f"sft_{split}_convs"] = len(part)
        lens = [len(tok.encode_chat(c)[0]) for c in convs[:2000]]
        print(f"Chat: {meta['sft_train_tokens']:,} tokens de treino, {meta['sft_val_tokens']:,} de validação "
              f"(conversa média {np.mean(lens):.0f} tokens, máx {max(lens)})")
        seen = np.bincount(np.fromfile(out / "sft_train.bin", dtype=np.uint16), minlength=tok.vocab_size)
        print(f"Tokens do vocabulário que nunca aparecem no treino: {(seen == 0).sum()} de {tok.vocab_size}")
        # vocabulário fixo de cada ferramenta, para a cópia forçada do agente (copia.py)
        fixas = fixas_das_conversas(convs)
        (out / "fixas.json").write_text(json.dumps(fixas, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Vocabulário fixo das ferramentas: " + ", ".join(f"{t} ({len(ws)})" for t, ws in fixas.items()))

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Pronto. Tudo em {out}/")


if __name__ == "__main__":
    main()
