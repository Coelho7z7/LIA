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
from pathlib import Path

import numpy as np

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
    corpus = "\n".join(texts + [m["content"] for c in convs for m in c])
    tok = BPETokenizer()
    print(f"Treinando tokenizador (vocab={args.vocab})...")
    tok.train(corpus, args.vocab)
    tok.save(out / "tokenizer.json")
    print(f"  {tok.vocab_size} tokens em {time.time() - t0:.1f}s")
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
        for split, part in (("train", convs[n_val:]), ("val", convs[:n_val])):
            ids, mask, starts = [], [], []
            for c in part:
                i, m = tok.encode_chat(c)
                starts.append(len(ids))
                ids += i
                mask += m
            np.array(ids, dtype=np.uint16).tofile(out / f"sft_{split}.bin")
            np.array(mask, dtype=np.uint8).tofile(out / f"sft_{split}_mask.bin")
            np.save(out / f"sft_{split}_starts.npy", np.array(starts, dtype=np.int64))
            meta[f"sft_{split}_tokens"] = len(ids)
            meta[f"sft_{split}_convs"] = len(part)
        lens = [len(tok.encode_chat(c)[0]) for c in convs[:2000]]
        print(f"Chat: {meta['sft_train_tokens']:,} tokens de treino, {meta['sft_val_tokens']:,} de validação "
              f"(conversa média {np.mean(lens):.0f} tokens, máx {max(lens)})")

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Pronto. Tudo em {out}/")


if __name__ == "__main__":
    main()
