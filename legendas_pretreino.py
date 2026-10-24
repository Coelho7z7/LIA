"""
Pré-treino extra com legendas de filmes e séries: a Lia lê conversa de verdade (gíria, resposta curta, "cê", ironia),
que a Wikipédia não tem. Parte de um modelo já pré-treinado na Wikipédia e mistura um pouco dela para não esquecer.

Fonte: OpenSubtitles v2018 (OPUS, Universidade de Helsinque), português do Brasil, só texto:
    data_legendas/pt_br.txt.gz   (object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/pt_br.txt.gz)

    python legendas_pretreino.py limpar --out out_g88l              # -> data_legendas/limpo_*.jsonl (amostra limpa)
    python legendas_pretreino.py codificar --out out_g88l --wiki out_v15 --wiki-tokens 100000000
    python train.py --stage pretrain --data out_g88l --out out_g88l --init out_g88/base.pt --preset large ...

Limpeza (o que os trabalhos com OpenSubtitles costumam tirar): travessão de diálogo, notas de música, anotação para
surdos ([risos], (suspira)), "NOME:" na frente da fala, créditos de quem legendou, "Anteriormente em...", linha sem
letras, e falas que se repetem demais ("Obrigado." aparece milhões de vezes).
"""

import argparse
import gzip
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

PASTA = Path(__file__).resolve().parent / "data_legendas"
ORIGEM = PASTA / "pt_br.txt.gz"

_CREDITO = re.compile(r"\b(legendas?|sincronia|sincronizad[oa]|tradu[cç][aã]o|traduzid[oa]|revis[aã]o|resinc\w*|"
                      r"ripad[oa]|adapta[cç][aã]o)\s*(:|por\b|e\b|feita)|www\.|https?:|\.com\b|\.org\b|\.net\b|"
                      r"opensubtitles|subtitle|insubs|legendas\.tv|anteriormente em", re.I)
_ANOTACAO = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>")
_MUSICA = re.compile(r"[♪♫#¶*]")
_NOME = re.compile(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ .'-]{1,24}:\s*")
_TRAVESSAO = re.compile(r"^\s*[-–—]+\s*")
_LETRAS = re.compile(r"[^\W\d_]")


def limpar_fala(t):
    if _CREDITO.search(t):
        return ""
    t = _ANOTACAO.sub(" ", t)
    t = _MUSICA.sub(" ", t)
    t = _NOME.sub("", t.strip())
    # "- Oi. - Tchau." vira duas falas
    partes = [_TRAVESSAO.sub("", p) for p in re.split(r"\s+[-–—]\s+(?=\S)", t)]
    out = []
    for p in partes:
        p = re.sub(r"\s+", " ", p).strip()
        if len(_LETRAS.findall(p)) < 2 or len(_LETRAS.findall(p)) < 0.5 * len(p):
            continue
        if p.isupper() and len(p) > 6:  # letreiro de tela ("NOVA YORK, 1985")
            continue
        out.append(p)
    return "\n".join(out)


def blocos(args):
    """Lê o .gz em ordem e devolve blocos de args.falas linhas seguidas, ainda sujas (cenas inteiras ensinam ida e
    volta). A limpeza fica para depois do sorteio: limpar 250 milhões de linhas para guardar 15% seria desperdício."""
    bloco = []
    with gzip.open(ORIGEM, "rt", encoding="utf-8", errors="replace") as f:
        for linha in f:
            bloco.append(linha)
            if len(bloco) >= args.falas:
                yield bloco
                bloco = []
    if bloco:
        yield bloco


def limpar(args):
    rng = random.Random(0)
    vistas = np.zeros(1 << 24, dtype=np.uint32)  # contagem por hash: um dicionário de falas não caberia na RAM
    arquivos = [open(PASTA / f"limpo_{i}.jsonl", "w", encoding="utf-8") for i in range(args.processos)]
    n_blocos = n_chars = lidos = cortadas = 0
    t0 = time.time()
    for bloco in blocos(args):
        lidos += 1
        if rng.random() >= args.amostra:
            continue
        falas = []
        for p in (p for linha in bloco for p in limpar_fala(linha).split("\n") if p):
            if len(p) <= 30:
                k = hash(p.lower()) & ((1 << 24) - 1)
                vistas[k] += 1
                if vistas[k] > args.max_repete:
                    cortadas += 1
                    continue
            if falas and falas[-1] == p:  # fala repetida em seguida (erro de sincronia)
                continue
            falas.append(p)
        if len(falas) < args.falas // 2:
            continue
        texto = "\n".join(falas)
        arquivos[n_blocos % len(arquivos)].write(json.dumps(texto, ensure_ascii=False) + "\n")
        n_blocos += 1
        n_chars += len(texto)
        if n_blocos % 200_000 == 0:
            print(f"  {n_blocos:,} blocos, {n_chars / 1e6:,.0f}M caracteres ({time.time() - t0:.0f}s)", flush=True)
        if n_chars >= args.chars:
            break
    for a in arquivos:
        a.close()
    print(f"Limpeza: {n_blocos:,} blocos de {lidos:,} lidos, {n_chars:,} caracteres, {cortadas:,} falas repetidas "
          f"cortadas em {time.time() - t0:.0f}s")


def _codificar(job):
    origem, tok_path, destino = job
    tok = BPETokenizer.load(tok_path)
    buf, n = array("H"), 0
    with open(destino, "wb") as f:
        for linha in open(origem, encoding="utf-8"):
            buf.append(tok.bos)
            buf.extend(tok.encode(json.loads(linha)))
            if len(buf) >= 1 << 22:
                f.write(buf.tobytes())
                n += len(buf)
                buf = array("H")
            if len(tok._cache) > 400_000:
                tok._cache.clear()
        f.write(buf.tobytes())
        n += len(buf)
    return origem.name, n


def codificar(args):
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    wiki = Path(args.wiki)
    shutil.copy(wiki / "tokenizer.json", out / "tokenizer.json")
    origens = sorted(PASTA.glob("limpo_*.jsonl"))
    jobs = [(o, out / "tokenizer.json", out / f"legendas_parte_{i}.bin") for i, o in enumerate(origens)]
    t0 = time.time()
    with Pool(min(args.processos, len(jobs))) as pool:
        for nome, n in pool.imap_unordered(_codificar, jobs):
            print(f"  {nome}: {n:,} tokens ({time.time() - t0:.0f}s)", flush=True)
    partes = [j[2] for j in jobs]
    ultima = np.fromfile(partes[-1], dtype=np.uint16)
    n_val = min(len(ultima) // 2, args.val_tokens)
    # validação: metade legendas, metade Wikipédia (para ver se ela esquece a Wikipédia)
    wiki_val = np.memmap(wiki / "pretrain_val.bin", dtype=np.uint16, mode="r")[:n_val]
    np.concatenate([ultima[:n_val], wiki_val]).tofile(out / "pretrain_val.bin")
    ultima[:n_val].tofile(out / "pretrain_val_legendas.bin")
    np.asarray(wiki_val).tofile(out / "pretrain_val_wiki.bin")
    # treino: legendas + pedaços sorteados da Wikipédia (o treino sorteia posições, então a ordem não importa)
    rng = np.random.default_rng(0)
    wiki_train = np.memmap(wiki / "pretrain_train.bin", dtype=np.uint16, mode="r")
    pedaco = 1 << 20
    n_ped = args.wiki_tokens // pedaco
    inicios = np.sort(rng.choice(len(wiki_train) // pedaco, n_ped, replace=False)) * pedaco
    n_leg = 0
    with open(out / "pretrain_train.bin", "wb") as f:
        for p in partes[:-1]:
            with open(p, "rb") as g:
                shutil.copyfileobj(g, f, 1 << 24)
            n_leg += p.stat().st_size // 2
        f.write(ultima[n_val:].tobytes())
        n_leg += len(ultima) - n_val
        for s in inicios:
            f.write(np.asarray(wiki_train[s:s + pedaco]).tobytes())
    del ultima
    for p in partes:
        p.unlink()
    n_train = (out / "pretrain_train.bin").stat().st_size // 2
    meta = {"pretrain_tokens": int(n_train + 2 * n_val), "legendas_tokens": int(n_leg),
            "wiki_tokens": int(n_ped * pedaco)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Pré-treino: {n_leg:,} tokens de legendas + {n_ped * pedaco:,} de Wikipédia "
          f"({n_ped * pedaco / n_train:.0%}) + {2 * n_val:,} de validação em {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["limpar", "codificar"])
    ap.add_argument("--out", default="out_g88l")
    ap.add_argument("--wiki", default="out_v15", help="pasta com tokenizer.json e pretrain_{train,val}.bin da Wikipédia")
    ap.add_argument("--wiki-tokens", type=int, default=100_000_000)
    ap.add_argument("--chars", type=int, default=1_000_000_000, help="quanto texto de legenda guardar")
    ap.add_argument("--amostra", type=float, default=0.15, help="fração dos blocos guardada (espalha por mais filmes)")
    ap.add_argument("--falas", type=int, default=40, help="falas por bloco")
    ap.add_argument("--max-repete", type=int, default=3000, help="vezes que a mesma fala curta pode aparecer")
    ap.add_argument("--processos", type=int, default=4)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"limpar": limpar, "codificar": codificar}[args.etapa](args)


if __name__ == "__main__":
    main()
