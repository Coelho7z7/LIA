"""
Exporta um modelo treinado para a versão web (motor em JavaScript, webapp/engine/).

    python export_web.py --ckpt out_v9/chat.pt --out webapp/modelo
    python export_web.py --ckpt out_v9/chat.pt --out webapp/modelo --dtype i8   # metade do tamanho

Gera:
  modelo.json     config, tokenizador, vocabulário fixo da cópia forçada, ferramentas e o índice
                  de cada tensor (onde começa no binário, formato e forma)
  pesos.HASH.NN.bin  os pesos em pedaços de até 24 MiB (o Cloudflare Pages aceita até 25 MiB por arquivo)

Formatos: f16 (padrão, ~56 MB) ou i8 por linha (~28 MB, cada linha com sua escala em f32).
As normas (vetores pequenos) vão sempre em f32. O lm_head não vai: é o mesmo peso do embedding.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import tools
from copia import carregar_fixas

CHUNK = 24 * 1024 * 1024


def tensor_names(n_layer):
    names = ["tok_emb.weight"]
    for i in range(n_layer):
        names += [f"blocks.{i}.{n}.weight" for n in ("norm1", "attn.qkv", "attn.proj", "norm2",
                                                      "mlp.gate", "mlp.up", "mlp.proj")]
    return names + ["norm.weight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--out", default="webapp/modelo")
    ap.add_argument("--dtype", choices=["f16", "i8"], default="f16")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd, cfg = ck["model"], ck["config"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("pesos.*.bin"):
        old.unlink()

    blob, index = bytearray(), []
    for name in tensor_names(cfg["n_layer"]):
        w = sd[name].float().numpy()
        while len(blob) % 16:  # alinhamento: qualquer tipo pode virar view direto no JS
            blob.append(0)
        entry = {"name": name, "shape": list(w.shape), "offset": len(blob)}
        if w.ndim == 1:
            data, entry["dtype"] = w.astype("<f4").tobytes(), "f32"
        elif args.dtype == "f16":
            data, entry["dtype"] = w.astype("<f2").tobytes(), "f16"
        else:
            scale = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127.0
            q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
            data = q.tobytes()
            entry["dtype"], entry["scales"] = "i8", len(blob) + len(data)
            pad = (-len(data)) % 16
            data += b"\0" * pad + scale.astype("<f4").tobytes()
            entry["scales"] += pad
        blob += data
        index.append(entry)

    # o hash no nome muda a cada modelo: o navegador nunca mistura pedaço velho com índice novo
    tag = hashlib.sha256(blob).hexdigest()[:10]
    chunks = []
    for k in range(0, len(blob), CHUNK):
        fname = f"pesos.{tag}.{k // CHUNK:02d}.bin"
        (out / fname).write_bytes(blob[k:k + CHUNK])
        chunks.append(fname)

    fixas = carregar_fixas(args.ckpt)
    manifest = {
        "versao": 1, "origem": str(args.ckpt), "passo": ck["iter"], "val_loss": ck["best_val"],
        "config": cfg, "dtype": args.dtype, "bytes": len(blob), "chunks": chunks, "tensores": index,
        "tokenizer": ck["tokenizer"], "fixas": fixas or {},
        "ferramentas": {n: {"confirm": t.confirm, "desc": t.desc} for n, t in tools.TOOLS.items()},
        "params": sum(int(np.prod(t["shape"])) for t in index),
    }
    (out / "modelo.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"{out}/modelo.json + {len(chunks)} pedaço(s), {len(blob) / 2**20:.1f} MiB ({args.dtype}), "
          f"cópia forçada: {'sim' if fixas else 'não'}")


if __name__ == "__main__":
    main()
