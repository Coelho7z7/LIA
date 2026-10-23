"""
Pré-treino com a Wikipédia em português: a Lia lê texto de verdade antes de aprender a conversar.

Fonte: wikimedia/wikipedia (20231101.pt) no Hugging Face, só texto (sem modelo de ninguém), em
data_wiki/parquet/. O prepare.py carregaria tudo na memória; aqui o texto é lido aos pedaços e
codificado em paralelo, um processo por arquivo.

    python wiki_pretreino.py tokenizador --conversas data_v14/conversas --out out_v15
    python prepare.py --conversas data_v14/conversas --out out_v15 --bpe-dropout 0.1 --tokenizer out_v15/tokenizer.json
    python wiki_pretreino.py codificar --out out_v15          # -> out_v15/pretrain_{train,val}.bin
    python train.py --stage pretrain --data out_v15 --out out_v15 --preset medium ...    # -> base.pt
    python train.py --stage sft --data out_v15 --out out_v15 ...                         # parte do base.pt

O tokenizador é treinado com partes iguais de Wikipédia e de conversas, para servir aos dois.
"""

import argparse
import json
import random
import re
import shutil
import sys
import time
from array import array
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tokenizer import BPETokenizer

PARQUET = Path(__file__).resolve().parent / "data_wiki" / "parquet"
# Seções do fim que são só listas e referências
_FIM = re.compile(r"\n(Referências|Ver também|Ligações externas|Bibliografia|Notas|Notas e referências|Leitura adicional|"
                  r"Fontes|Referências bibliográficas)\s*\n")


def artigos(arquivo):
    import pyarrow.parquet as pq
    for lote in pq.ParquetFile(arquivo).iter_batches(columns=["text"], batch_size=500):
        for t in lote.column("text").to_pylist():
            t = limpar(t or "")
            if t:
                yield t


def limpar(t):
    m = _FIM.search(t)
    if m:
        t = t[:m.start()]
    # sobras de predefinições removidas na extração: "()", "(; )", "de 1576 a , data"
    t = re.sub(r"\s*\(\s*[,;:]?\s*\)", "", t)
    t = re.sub(r" +([,.;:])", r"\1", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t if len(t) >= 300 else ""  # esboço de uma linha ensina pouco


def tokenizador(args):
    rng = random.Random(0)
    wiki, n = [], 0
    arquivos = sorted(PARQUET.glob("*.parquet"))
    for a in arquivos:  # um pouco de cada arquivo, sorteado
        for t in artigos(a):
            if rng.random() < 0.05:
                wiki.append(t)
                n += len(t)
            if n >= args.chars * (arquivos.index(a) + 1) / len(arquivos):
                break
    conv, m = [], 0
    for f in sorted(Path(args.conversas).rglob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            msgs = [x["content"] for x in json.loads(line)["messages"]]
            conv.extend(msgs)
            m += sum(map(len, msgs))
            if m >= args.chars:
                break
        if m >= args.chars:
            break
    corpus = "\n".join(wiki + conv)
    print(f"Tokenizador: {sum(map(len, wiki)):,} caracteres de Wikipédia + {sum(map(len, conv)):,} de conversas")
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(corpus, args.vocab)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tok.save(out / "tokenizer.json")
    print(f"  {tok.vocab_size} tokens em {time.time() - t0:.0f}s -> {out / 'tokenizer.json'}")


def _codificar(job):
    arquivo, tok_path, destino = job
    tok = BPETokenizer.load(tok_path)
    buf, n, docs = array("H"), 0, 0
    with open(destino, "wb") as f:
        for t in artigos(arquivo):
            buf.append(tok.bos)
            buf.extend(tok.encode(t))
            docs += 1
            if len(buf) >= 1 << 22:
                f.write(buf.tobytes())
                n += len(buf)
                buf = array("H")
            if len(tok._cache) > 400_000:  # o cache de palavras cresce sem fim com nomes próprios
                tok._cache.clear()
        f.write(buf.tobytes())
        n += len(buf)
    return arquivo.name, n, docs


def codificar(args):
    out = Path(args.out)
    tok_path = out / "tokenizer.json"
    arquivos = sorted(PARQUET.glob("*.parquet"))
    jobs = [(a, tok_path, out / f"pretrain_parte_{i}.bin") for i, a in enumerate(arquivos)]
    t0 = time.time()
    with Pool(min(args.processos, len(jobs))) as pool:
        for nome, n, docs in pool.imap_unordered(_codificar, jobs):
            print(f"  {nome}: {docs:,} artigos, {n:,} tokens ({time.time() - t0:.0f}s)", flush=True)
    # validação: o começo da última parte; treino: todo o resto
    partes = [j[2] for j in jobs]
    ultima = np.fromfile(partes[-1], dtype=np.uint16)
    n_val = min(len(ultima) // 2, args.val_tokens)
    ultima[:n_val].tofile(out / "pretrain_val.bin")
    with open(out / "pretrain_train.bin", "wb") as f:
        for p in partes[:-1]:
            with open(p, "rb") as g:
                shutil.copyfileobj(g, f, 1 << 24)
        f.write(ultima[n_val:].tobytes())
    del ultima
    for p in partes:
        p.unlink()
    n_train = (out / "pretrain_train.bin").stat().st_size // 2
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["pretrain_tokens"] = int(n_train + n_val)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Pré-treino: {n_train:,} tokens de treino + {n_val:,} de validação em {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["tokenizador", "codificar"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--conversas", default="data/conversas")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--chars", type=int, default=40_000_000, help="caracteres de cada lado (Wikipédia e conversas)")
    ap.add_argument("--processos", type=int, default=6)
    ap.add_argument("--val-tokens", type=int, default=4_000_000)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"tokenizador": tokenizador, "codificar": codificar}[args.etapa](args)


if __name__ == "__main__":
    main()
